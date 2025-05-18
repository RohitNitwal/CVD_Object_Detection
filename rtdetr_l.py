from ultralytics import YOLO

def train_rtdetr_simplified():
    
    model = YOLO('rtdetr-l.pt')

    training_args = dict(
        data='/mnt/combined/rohit_nitwal/cvd2_dataset.yaml',
        epochs=200,
        imgsz=640,
        batch=27,
        device='0',
        workers=10,
        project='/mnt/combined/rohit_nitwla/results/rtdetr/',
        name='rtdetr',
        exist_ok=True,      
        patience=20,     
        save_period=-1,     
        optimizer='AdamW',  
        seed=0,            
        deterministic=True, 
        cos_lr=False,       
        close_mosaic=10,   
        # amp=True,         
        iou=0.7,
        plots=True,     
        augment=False,      
        nms=False,          
        lr0=0.0002,         
        lrf=0.01,          
        weight_decay=0.05,  
        warmup_epochs=5,    
        translate=0.1,      
        scale=0.5,        
        fliplr=0.5,        
        auto_augment='randaugment', 
        erasing=0.4,        
    )


    results = model.train(**training_args)

    print(f"Training complete. Results saved in: {results.save_dir}")
    print(f"Best model weights: {results.best}")

if __name__ == '__main__':
    train_rtdetr_simplified()
