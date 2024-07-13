```markdown
# YOLOv5 Oil Storage Tanks Detection

This repository contains the implementation for training a YOLOv5 model to detect different types of oil storage tanks in satellite images. 

## Prerequisites

Ensure you have NVIDIA CUDA installed:
```bash
!nvcc -V
```

Clone the YOLOv5 repository:
```bash
!git clone https://github.com/ultralytics/yolov5.git
```

Install the required dependencies:
```bash
# !pip install -qr yolov5/requirements.txt  # Uncomment to install all dependencies
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset Preparation

1. **Define the path to the dataset:**
    ```python
    DATA_PATH = '/kaggle/input/oil-storage-tanks/Oil Tanks/'
    ```

2. **List files and directories in the specified path:**
    ```python
    os.listdir(DATA_PATH)
    ```

3. **Function to convert bounding box data:**
    ```python
    def conv_bbox(box_dict):
        xs = np.array(list(set([i['x'] for i in box_dict])))
        ys = np.array(list(set([i['y'] for i in box_dict])))
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        return y_min, x_min, y_max, x_max
    ```

4. **Setup directories for training and testing images:**
    ```python
    source = os.path.join(DATA_PATH, 'image_patches')
    destination_1 = 'train'
    destination_2 = 'test'

    if not os.path.isdir(destination_1):
        os.mkdir(destination_1)
    if not os.path.isdir(destination_2):
        os.mkdir(destination_2)
    ```

5. **Load and process annotations:**
    ```python
    label_to_num = {'Tank': 0, 'Tank Cluster': 1, 'Floating Head Tank': 2}
    annotations = []
    json_labels = json.load(open(os.path.join(DATA_PATH, 'labels.json')))
    for i in tqdm(range(len(json_labels))):
        file = json_labels[i]['file_name']
        if file.startswith('01'):
            shutil.copy(source + '/' + file, destination_2)
        elif json_labels[i]['label'] != 'Skip':
            shutil.copy(source + '/' + file, destination_1)
            for label in json_labels[i]['label'].keys():
                for box in json_labels[i]['label'][label]:
                    y_min, x_min, y_max, x_max = conv_bbox(box['geometry'])
                    width = x_max - x_min
                    height = y_max - y_min
                    annotations.append((file.split('.')[0], label_to_num[label], label, [x_min, y_min, width, height]))
    annotations = pd.DataFrame(annotations, columns=['image_name', 'class', 'class_name', 'bbox'])
    ```

6. **Split dataset into training and validation sets:**
    ```python
    df_train, df_valid = model_selection.train_test_split(
        annotations, 
        test_size=0.1, 
        random_state=42, 
        shuffle=True, 
        stratify=annotations['class']
    )
    ```

## Data Conversion for YOLO

1. **Convert data to YOLO format:**
    ```python
    def convert(data, data_type):
        df = data.groupby('image_name')['bbox'].apply(list).reset_index(name='bboxes')
        df['classes'] = data.groupby('image_name')['class'].apply(list).reset_index(drop=True)
        df.to_csv(data_type + '.csv', index=False)
    df_train = convert(df_train, 'train')
    df_valid = convert(df_valid, 'validation')
    ```

2. **Setup directories for YOLO data:**
    ```python
    %cd yolov5
    !mkdir tank_data
    %cd tank_data
    !mkdir images labels
    %cd images
    !mkdir train validation
    %cd ../labels
    !mkdir train validation
    %cd ../..
    ```

3. **Process data and save in YOLO format:**
    ```python
    def process_data(data, data_type='train'):
        for _, row in tqdm(data.iterrows(), total=len(data)):
            image_name = row['image_name']
            bounding_boxes = row['bboxes']
            classes = row['classes']
            yolo_data = []
            for bbox, Class in zip(bounding_boxes, classes):
                x = bbox[0]
                y = bbox[1]
                w = bbox[2]
                h = bbox[3]
                x_center = x + w / 2
                y_center = y + h / 2
                x_center /= 512
                y_center /= 512
                w /= 512
                h /= 512
                yolo_data.append([Class, x_center, y_center, w, h])
            yolo_data = np.array(yolo_data)
            np.savetxt(
                os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
                yolo_data,
                fmt=["%d", "%f", "%f", "%f", "%f"]
            )
            shutil.copyfile(
                os.path.join(INPUT_PATH, f"train/{image_name}.jpg"),
                os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg")
            )
    df_train = pd.read_csv('/kaggle/working/train.csv')
    df_train.bboxes = df_train.bboxes.apply(ast.literal_eval)
    df_train.classes = df_train.classes.apply(ast.literal_eval)
    df_valid = pd.read_csv('/kaggle/working/validation.csv')
    df_valid.bboxes = df_valid.bboxes.apply(ast.literal_eval)
    df_valid.classes = df_valid.classes.apply(ast.literal_eval)
    process_data(df_train, data_type='train')
    process_data(df_valid, data_type='validation')
    ```

## Training

1. **Create YOLO configuration file:**
    ```python
    %cd yolov5
    %%writefile tank.yaml
    train: tank_data/images/train
    val: tank_data/images/validation
    nc: 3
    names: ['Tank', 'Tank Cluster', 'Floating Head Tank']
    ```

2. **Start training:**
    ```bash
    !python train.py --img 512 --batch 16 --epochs 200 --data tank.yaml --cfg models/yolov5l.yaml --name oiltank
    ```

## Visualizing Predictions

1. **Define a function to plot bounding boxes:**
    ```python
    def plot_BBox(img_name, ax):
        sns.set({'figure.figsize': (20, 10)})
        img_path = os.path.join(path + 'test', img_name)
        image = vision.open_image(img_path)
        image.show(ax=ax, title='Ground Truth ' + img_name)
        no, row, col = map(int, img_name.split('.')[0].split('_'))
        img_id = (no - 1) * 100 + row * 10 + col
        idx = -1
        bboxes = []
        labels = []
        classes = []
        if json_labels[img_id]['label'] != 'Skip':
            for label in json_labels[img_id]['label'].keys():
                for box in json_labels[img_id]['label'][label]:
                    bboxes.append(conv_bbox(box['geometry']))
                    classes.append(label)
            labels = list(range(len(classes)))
            idx = 1
        if idx != -1:
            BBox = vision.ImageBBox.create(*image.size, bboxes, labels, classes)
            image.show(y=BBox, ax=ax)
    ```

2. **Display images with ground truth and predicted bounding boxes:**
    ```python
    sns.set({'figure.figsize': (20, 30 * 10)})
    fig, ax = plt.subplots(30, 2)
    for i, img_f in enumerate(sorted(os.listdir('/kaggle/input/oil-storage-tanks/Oil Tanks/image_patches/'))[40:70]):
        image = vision.open_image('/kaggle/input/oil-storage-tanks/Oil Tanks/image_patches/' + img_f)
        image.show(ax=ax[i][0], title='Predicted ' + img_f)
        plot_BBox(img_f, ax[i][1])
    plt.show()
    ```
```
