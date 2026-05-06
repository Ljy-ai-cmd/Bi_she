import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, Imagenet1k, VastTrack
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import VisEvent, LasHeR, DepthTrack
from lib.train.dataset import Otb99_lang, Tnl2k, RefCOCOSeq
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process

def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': getattr(cfg.DATA.TEMPLATE, "FACTOR", None),
                                   'search': getattr(cfg.DATA.SEARCH, "FACTOR", None)}
    settings.output_sz = {'template': getattr(cfg.DATA.TEMPLATE, "SIZE", 128),
                          'search': getattr(cfg.DATA.SEARCH, "SIZE", 256)}
    settings.center_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "CENTER_JITTER", None),
                                     'search':getattr(cfg.DATA.SEARCH, "CENTER_JITTER", None)}
    settings.scale_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "SCALE_JITTER", None),
                                    'search': getattr(cfg.DATA.SEARCH, "SCALE_JITTER", None)}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.multi_modal_vision = getattr(cfg.DATA, "MULTI_MODAL_VISION", False)
    settings.multi_modal_language = getattr(cfg.DATA, "MULTI_MODAL_LANGUAGE", False)
    settings.infrared_only = getattr(cfg.DATA, "INFRARED_ONLY", False)
    settings.use_nlp = cfg.DATA.USE_NLP
    settings.accum_iter = getattr(cfg.TRAIN, "ACCUM_ITER", 1)
    settings.save_best = getattr(cfg.TRAIN, "SAVE_BEST", True)
    settings.save_every_epoch = getattr(cfg.TRAIN, "SAVE_EVERY_EPOCH", False)
    # LAST-ViT: 传递冻结epoch配置
    settings.FREEZE_BASE_EPOCHS = getattr(cfg.TRAIN, "FREEZE_BASE_EPOCHS", 0)
    settings.UNFREEZE_STRATEGY = getattr(cfg.TRAIN, "UNFREEZE_STRATEGY", "gradual")
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        settings.fix_norm = True
    else:
        settings.fix_norm = False


def names2datasets(name_list: list, settings, image_loader, split_override=None):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full",
                        "COCO17", "VID", "TRACKINGNET", "IMAGENET1K",
                        "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train","LasHeR_val", "VisEvent",
                        "REFCOCOG", "TNL2K_train", "OTB99_train","VASTTRACK", "ANTIUAV", "ANTIUAV_CLAHE"]  # 添加ANTIUAV及其CLAHE版本
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader,
                                           multi_modal_vision=settings.multi_modal_vision,
                                           multi_modal_language=settings.multi_modal_language,
                                           use_nlp=settings.use_nlp['LASOT']))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language,
                                      use_nlp=settings.use_nlp['LASOT']))
        if name == "VASTTRACK":
            datasets.append(VastTrack(settings.env.vasttrack_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language,
                                      use_nlp=settings.use_nlp['VASTTRACK']))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['GOT10K']
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['GOT10K']
                                       ))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['GOT10K']
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['GOT10K']
                                       ))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['GOT10K']
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['GOT10K']
                                       ))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader,
                                               multi_modal_vision=settings.multi_modal_vision,
                                               multi_modal_language=settings.multi_modal_language,
                                               use_nlp=settings.use_nlp['COCO']
                                               ))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader,
                                          multi_modal_vision=settings.multi_modal_vision,
                                          multi_modal_language=settings.multi_modal_language,
                                          use_nlp=settings.use_nlp['COCO']
                                          ))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader,
                                                 multi_modal_vision=settings.multi_modal_vision,
                                                 multi_modal_language=settings.multi_modal_language,
                                                 use_nlp=settings.use_nlp['TRACKINGNET']
                                                 ))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language,
                                            use_nlp=settings.use_nlp['TRACKINGNET']
                                            ))
        if name == "IMAGENET1K":
            datasets.append(Imagenet1k(settings.env.imagenet1k_dir, image_loader=image_loader))
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir,
                                       dtype='color' if not settings.multi_modal_vision else 'rgbcolormap',
                                       split='train',
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['DEPTHTRACK']
                                       ))
        if name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir,
                                       dtype='color' if not settings.multi_modal_vision else 'rgbcolormap',
                                       split='val',
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['DEPTHTRACK']
                                       ))
        if name == "LasHeR_all":
            datasets.append(LasHeR(settings.env.lasher_dir,
                                   dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                   split='all',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language,
                                   use_nlp=settings.use_nlp['LASHER']
                                   ))
        if name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_dir,
                                   dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                   split='train',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language,
                                   use_nlp=settings.use_nlp['LASHER']
                                   ))
        if name == "LasHeR_val":
            datasets.append(LasHeR(settings.env.lasher_dir,
                                   dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                   split='val',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language,
                                   use_nlp=settings.use_nlp['LASHER']
                                   ))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir,
                                     dtype='color' if not settings.multi_modal_vision else 'rgbrgb',
                                     split='train',
                                     multi_modal_vision=settings.multi_modal_vision,
                                     multi_modal_language=settings.multi_modal_language,
                                     use_nlp=settings.use_nlp['VISEVENT']
                                     ))
        if name == "REFCOCOG":
            datasets.append(RefCOCOSeq(settings.env.refcoco_dir, split="train", image_loader=image_loader,
                                       name="refcocog", splitBy="google",
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['REFCOCOG']
                                       ))
        if name == "TNL2K_train":
            datasets.append(Tnl2k(settings.env.tnl2k_dir, split=None, image_loader=image_loader,
                                  multi_modal_vision=settings.multi_modal_vision,
                                  multi_modal_language=settings.multi_modal_language,
                                  use_nlp=settings.use_nlp['TNL2K']
                                  ))
        elif name == "OTB99_train":
            datasets.append(Otb99_lang(settings.env.otb99_dir, split='train', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['OTB99']
                                       ))
        elif name == "ANTIUAV":
            from lib.train.dataset.antiuav import AntiUAVDataset
            split = split_override if split_override is not None else 'train'
            datasets.append(AntiUAVDataset(settings.env.antiuav_dir, split=split, image_loader=image_loader,
                                           multi_modal_vision=settings.multi_modal_vision,
                                           multi_modal_language=settings.multi_modal_language,
                                           use_nlp=settings.use_nlp.get('ANTIUAV', False),
                                           infrared_only=settings.infrared_only))
        
        elif name == "ANTIUAV_CLAHE":  # 添加CLAHE增强版本
            from lib.train.dataset.antiuav_clahe import AntiUAVDatasetCLAHE
            split = split_override if split_override is not None else 'train'
            
            # 从settings读取CLAHE参数
            use_clahe = getattr(settings, 'use_clahe', True)
            clahe_clip_limit = getattr(settings, 'clahe_clip_limit', 2.0)
            clahe_grid_size = getattr(settings, 'clahe_grid_size', (8, 8))
            
            datasets.append(AntiUAVDatasetCLAHE(
                settings.env.antiuav_dir, 
                split=split, 
                image_loader=image_loader,
                multi_modal_vision=settings.multi_modal_vision,
                multi_modal_language=settings.multi_modal_language,
                use_nlp=settings.use_nlp.get('ANTIUAV', False),
                use_clahe=use_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_grid_size=tuple(clahe_grid_size)
            ))

    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    
    # 设置CLAHE预处理参数
    settings.use_clahe = getattr(cfg.DATA, 'USE_CLAHE', False)
    settings.clahe_clip_limit = getattr(cfg.DATA, 'CLAHE_CLIP_LIMIT', 2.0)
    settings.clahe_grid_size = tuple(getattr(cfg.DATA, 'CLAHE_GRID_SIZE', [8, 8]))
    
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqTrackProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          multi_modal_language=settings.multi_modal_language,
                                                          settings=settings)

    # Train sampler and loader
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    # print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode,
                                            multi_modal_language=settings.multi_modal_language
                                            )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    prefetch_factor = getattr(cfg.TRAIN, "PREFETCH_FACTOR", 4)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler,
                             pin_memory=True, prefetch_factor=prefetch_factor)
    loaders = [loader_train]

    if hasattr(cfg.DATA, "VAL") and hasattr(cfg.DATA.VAL, "DATASETS_NAME"):
        val_samples = getattr(cfg.DATA.VAL, "SAMPLE_PER_EPOCH", max(1000, cfg.DATA.TRAIN.SAMPLE_PER_EPOCH // 10))
        val_ratios = getattr(cfg.DATA.VAL, "DATASETS_RATIO", [1 for _ in cfg.DATA.VAL.DATASETS_NAME])
        val_epoch_interval = getattr(cfg.DATA.VAL, "EPOCH_INTERVAL", 1)

        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader, split_override='val'),
                                              p_datasets=val_ratios,
                                              samples_per_epoch=val_samples,
                                              max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                              num_template_frames=settings.num_template, processing=data_processing_train,
                                              frame_sample_mode=sampler_mode,
                                              multi_modal_language=settings.multi_modal_language
                                              )
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                               num_workers=cfg.TRAIN.NUM_WORKER, drop_last=False, stack_dim=1, sampler=val_sampler,
                               pin_memory=True, epoch_interval=val_epoch_interval, prefetch_factor=prefetch_factor)
        loaders.append(loader_val)

    return loaders


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    
    # LAST-ViT: 检查是否需要冻结基础编码器
    freeze_base_epochs = getattr(cfg.TRAIN, "FREEZE_BASE_EPOCHS", 0)
    if freeze_base_epochs > 0 and is_main_process():
        print(f"\n{'='*60}")
        print(f"LAST-ViT: Freezing base encoder for first {freeze_base_epochs} epochs")
        print(f"{'='*60}\n")
    
    if train_type == "peft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad]},
        ]
        for n, p in net.named_parameters():
            if "prompt" not in n:
                p.requires_grad = False

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "fft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR / cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "text_frozen":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "encoder" in n and "clip" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]
        for n, p in net.named_parameters():
            if ("clip" in n) or ("bert" in n):
                p.requires_grad = False
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    else:
        # LAST-ViT: 如果设置了 FREEZE_BASE_EPOCHS，冻结基础编码器
        if freeze_base_epochs > 0:
            # 冻结编码器参数
            for n, p in net.named_parameters():
                if "encoder" in n:
                    # 检查是否是频域选择器参数（不冻结）
                    if "freq_selector" in n or "importance_mlp" in n:
                        p.requires_grad = True  # 频域选择器需要训练
                    else:
                        p.requires_grad = False  # 冻结基础编码器
            
            # 只优化频域选择器和解码器
            param_dicts = [
                # 解码器参数
                {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
                # 频域选择器参数（使用正常学习率）
                {
                    "params": [p for n, p in net.named_parameters() if "encoder" in n and ("freq_selector" in n or "importance_mlp" in n) and p.requires_grad],
                    "lr": cfg.TRAIN.LR,
                },
            ]
        else:
            param_dicts = [
                {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in net.named_parameters() if "encoder" in n and p.requires_grad],
                    "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
                },
            ]
        
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
