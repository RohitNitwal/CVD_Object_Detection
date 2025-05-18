from ultralytics import YOLO

def train_yolo12m_resume_simplified():

    model = YOLO('yolo12m.pt')  

    training_args = dict(
        data='/mnt/combined/rohit_nitwal/cvd2_dataset.yaml',
        epochs=50,         
        imgsz=640,
        batch=27,
        device='0',
        workers=10,
        project='/mnt/combined/rohit/NLP/results_ultralytics/yolo12m/',
        name='yolo12m_cvd2_balanced_resume',
        exist_ok=True,     
        patience=20,       
        save_period=-1,     
        optimizer='AdamW',  
        seed=0,            
        deterministic=True, 
        close_mosaic=10,    
        resume=False,      
        iou=0.7,           # NMS iou
        plots=True,        
        lr0=0.0002,         
        lrf=0.01,           
        weight_decay=0.05,  
        warmup_epochs=3.0, 
        box=0.7,            
        cls=1.5,         
        dfl=1.0,            
        translate=0.1,
        scale=0.5,
        auto_augment='randaugment',
        erasing=0.4,
    )

    results = model.train(**training_args)

    print(f"Training complete. Results saved in: {results.save_dir}")
    print(f"Best model weights: {results.best}")

if __name__ == '__main__':
    train_yolo12m_resume_simplified()
