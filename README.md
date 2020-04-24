# Caption-Detection
ABBY hometask

Datasets are available [here](https://drive.google.com/open?id=1VXpGMbfL-5qdzQN2z4bWsz21ojvnPdZv)

Example:
![image](./images/labelme_example.jpg)

Augmentations exmaple:

<img src="images/augmentation_example.png"  width="1072" height="1072" alt="image">

![image](images/augmentation_example.png)

File description:
- Experiment.ipynb: main experiment file
- annotations_utils.py: utils to annotate source dataset
- labelme_utils.py: utils to work with labelme format
- blending.py: functions to blend images(in order to change background)
- boxes_utils.py: utils to works with different boxes format
- post_processing.py: function to post-process model output masks to extract boxes and counters
- metrics.py: functions to calculate object detections metrics in order to evaluate model
- show_utils.py: functions to draw images
- image_utils.py: some image utils
