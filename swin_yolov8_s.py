import torch.nn as nn
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from ultralytics import YOLO

class SwinBackbone(nn.Module):
    """Swin-Base backbone for YOLOv8: returns P3, P4, P5 feature maps."""
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1, progress=True)
        return_nodes = {
            'features.2': 'P3',
            'features.3': 'P4',
            'features.4': 'P5'
        }
        self.body = create_feature_extractor(backbone, return_nodes)
        # project dims [256,512,1024] → [128,256,512] for YOLO-small
        self.proj = nn.ModuleList([
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024,512, 1),
        ])

    def forward(self, x):
        feats = self.body(x)
        return [
            self.proj[i](feats[n]) for i,n in enumerate(['P3','P4','P5'])
        ]

def freeze_early_layers(backbone: SwinBackbone):
    """Freeze patch-embed & stage1 layers for initial phase."""
    for name, p in backbone.body.named_parameters():
        if 'patch_embed' in name or 'layers.0' in name:
            p.requires_grad = False

def unfreeze_all(backbone: SwinBackbone):
    """Unfreeze all backbone parameters."""
    for p in backbone.body.parameters():
        p.requires_grad = True

if __name__ == '__main__':
    # 1) Initialize YOLOv8-small with the Swin backbone
    yolo   = YOLO('yolov8s.pt')
    swin   = SwinBackbone()
    yolo.model.model[0] = swin

    common_args = dict(
        data     = '/mnt/combined/rohit_nitwal/cvd2_dataset.yaml',
        imgsz    = 640,
        batch    = 12,
        device   = '0',
        project  = '/mnt/combined/rohit_nitwal/results/yolov8/',
        name     = 'swin_yolov8',
        patience = 20,
        amp      = True,    
        mosaic   = 1.0,     
        mixup    = 0.5,      
        cache    = False,   
        workers  = 4,        
        plots    = True,     
        verbose  = True,
    )

    # ── Phase 1: Freeze low-level Swin for 10 epochs ──
    freeze_early_layers(swin)
    yolo.train(
        epochs = 10,
        lr0    = 5e-4,
        cos_lr = True,
        **common_args
    )

    # ── Phase 2: Unfreeze & continue to 200 epochs ──
    unfreeze_all(swin)
    yolo.train(
        epochs = 200,        # total epochs
        lr0    = 5e-4,
        cos_lr = True,
        # resume 
        **common_args
    )
