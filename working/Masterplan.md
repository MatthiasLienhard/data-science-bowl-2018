
# Nuclei Image Recognicion

## Data type Conversions
* images
    * numpy array[n,x,y,3], dtype=uint8
    * n images of dim x\*y pixels, 3 color intensities
* masks
    * numpy array[n,x,y], dtype=bool
    * False = Background, True = nuclei
* labeled masks
    * numpy array[n,x,y], dtype=uint8 
    * 0= background
    * 1, 2, 3, ... different nuclei
    
## OOP?
* data container class
    * get_images(): images
    * get_images_featrues(): data frame
    * get_truth(): masks
    * get_truth_labeled(): labled mask
    * get_pred(): masks()
    * get_pred_labeled(): masks()
    * write_submission("filename")
* model class?
    * Basti: prototype?
    * __init__ -> set parameters
    * fit(images, masks)
    * predict(images)
    
    
## Functions to be implemented:
* read_images(path) 
    * returns images and vector storing original size
    * all in same format
* read_masks(path)
    * returns masks
* read_labled_masks(path)
    * returns labeled masks
* classify_images(images)
    * returns data frame with image features
        * type: e.g. colored vs grayscale, different stainings
        * avg/min/max/sd nucleus size
        * heigth/width
        * number of touching nuclei
        * ...
* get_iou(truth, prediction)
    * unlabled masks
        * returns array of iou scores
    * labled masks
        * returns array of iou score lists (one per nucleus in truth)
* get_score(truth,predictions, th=None)
    * only on labled masks
    * if th is in [.5,1]:
        * returns array of fraction of recognized nuclei per image at threshold th
    * else 
        * array with average fraction of recognized nuclei per image at all thresholds np.arange(0.5,0.95,0.05)
        * the average over all image should correspond to the score used by kaggle
* show_image(images, masks, labled_masks, idx)
    * sanity check function (e.g. plot the three images) but also print score, rank, image classification, other features
    * idx can be "random", "worst", "bad", "average", "good","best"
        
    
    


```python

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-1-df6fdd0e565e> in <module>()
    ----> 1 range(0.5,0.95,0.05)
    

    TypeError: 'float' object cannot be interpreted as an integer

