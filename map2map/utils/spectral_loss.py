# map2map/utils/spectral_loss.py
import math
import torch

def _get_kgrid(size, L_sub, device):
    """
    size  : (Nx, Ny, Nz)
    L_sub : tamaño físico de la subcaja en [Mpc/h]
    """
    Nx, Ny, Nz = size
    dx = L_sub / Nx
    dy = L_sub / Ny
    dz = L_sub / Nz

    kx = torch.fft.fftfreq(Nx, d=dx) * 2.0 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy) * 2.0 * math.pi
    kz = torch.fft.fftfreq(Nz, d=dz) * 2.0 * math.pi

    kx = kx.to(device)
    ky = ky.to(device)
    kz = kz.to(device)

    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)   # (Nx,Ny,Nz)

    return k_mag.unsqueeze(0).unsqueeze(0)      # (1,1,Nx,Ny,Nz)


def power_spectrum_loss(eul_out, eul_tgt,
                        L_sub,
                        kmin=None,
                        alpha=2.0,
                        eps=1e-6):
    """
    eul_out, eul_tgt : (B, C, Nx, Ny, Nz)
    L_sub            : tamaño de la subcaja [Mpc/h]
    kmin             : solo penalizamos k >= kmin (en h/Mpc)
    alpha            : peso ~ (k/k_nyq)**alpha
    """
    device = eul_out.device
    _, _, Nx, Ny, Nz = eul_out.shape
    kgrid = _get_kgrid((Nx, Ny, Nz), L_sub, device)   # (1,1,Nx,Ny,Nz)

    # 1) quitar modo k=0 (densidad media)
    pred = eul_out - eul_out.mean(dim=(-3, -2, -1), keepdim=True)
    tgt  = eul_tgt  - eul_tgt.mean(dim=(-3, -2, -1), keepdim=True)

    # 2) FFT 3D
    pred_k = torch.fft.fftn(pred, dim=(-3, -2, -1))
    tgt_k  = torch.fft.fftn(tgt,  dim=(-3, -2, -1))

    # 3) potencia por modo
    P_pred = pred_k.abs()**2
    P_tgt  = tgt_k.abs()**2

    # 4) comparar en log P
    logP_pred = torch.log(P_pred + eps)
    logP_tgt  = torch.log(P_tgt  + eps)
    diff = logP_pred - logP_tgt   # (B,C,Nx,Ny,Nz)

    # 5) pesos en función de k
    k_mag = kgrid
    k_nyq = k_mag.max().clamp(min=eps)
    w = (k_mag / k_nyq)**alpha

    if kmin is not None:
        w = w * (k_mag >= kmin)

    # 6) loss = < w(k) [ΔlogP(k)]² >
    loss = (w * diff**2).mean()
    return loss