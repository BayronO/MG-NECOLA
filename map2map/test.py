import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast, lag2eul
from .utils import import_attr, load_model_state_dict


def test(args):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            warnings.warn('Not parallelized but given more than 1 GPUs')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda', 0)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        if args.num_threads is None:
            args.num_threads = int(os.environ['SLURM_CPUS_ON_NODE'])
        torch.set_num_threads(args.num_threads)

    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()

    # --- Dataset ---
    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    # --- Modelo ---
    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(sum(in_chan), sum(out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)

    criterion = import_attr(args.criterion, torch.nn, models,
                            callback_at=args.callback_at)()
    criterion.to(device)

    # --- Cargar checkpoint ---
    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    epoch_loaded = state.get('epoch', 'unknown')
    print(f'✅ Modelo cargado desde {args.load_state} (epoch {epoch_loaded})')
    del state

    suffix = f'_state{epoch_loaded}'
    model.eval()


    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, target = data['input'].to(device), data['target'].to(device)
            output = model(input)
            input, output, target = narrow_cast(input, output, target)

            lag_out = output
            lag_tgt = target

            eul_out, eul_tgt = lag2eul([lag_out, lag_tgt],
                **args.misc_kwargs
            )

            lag_loss = criterion(lag_out, lag_tgt)
            eul_loss = criterion(eul_out, eul_tgt)
            loss = torch.log(eul_loss * lag_loss**3.0)

            print(f"📦 Sample {i} | Lag loss: {lag_loss.item():.3e} | "
                  f"Eul loss: {eul_loss.item():.3e} | Total: {loss.item():.3e}")

            # --- Desnormalizar (output y target) ---
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm = import_attr(norm, norms, callback_at=args.callback_at)
                    norm(output[:, start:stop], undo=True, **args.misc_kwargs)
                    norm(target[:, start:stop], undo=True, **args.misc_kwargs)
                    start = stop
            # --- Ensamblar salida si se requiere ---
            test_dataset.assemble(f'_out{suffix}', out_chan, output, data['target_relpath'])
