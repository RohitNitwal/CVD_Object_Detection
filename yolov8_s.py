from ultralytics import YOLO

if __name__ == '__main__':
    
    # 1) Initialize a YOLOv8-small model
    model = YOLO('yolov8s.pt')  

    training_args = dict(
        data='/mnt/combined/rohit_nitwal/cvd2_dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=48,
        device='0',  
        project='/mnt/combined/rohit_nitwal/results/yolov8/',
        name='yolov8_s', 
        workers=8,
        exist_ok=True,
        amp=True,
        close_mosaic=10,
        warmup_epochs=3,
        lr0=0.01,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        dropout=0.0,
        patience=100,
        save_period=-1,
        plots=True,      
        verbose=True    
    )

   
    results = model.train(**training_args)

    print(f"Training complete. Best model saved at: {results.save_dir}")
    print(f"Best model weights: {results.best}")
