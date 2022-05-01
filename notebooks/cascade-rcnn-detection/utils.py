import numpy as np
import wandb


def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(round(_x, 5), round(_y, 5), format(round(value, 5), ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# Get the raw bounding box 
# Ref: https://www.kaggle.com/yujiariyasu/plot-3positive-classes
def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []  
            
    return bboxes

# Scale the bounding boxes.
def scale_bbox(img_size, row, bboxes):
    # Get scaling factor
    scale_x = img_size/row.dim1
    scale_y = img_size/row.dim0
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]*scale_x, 4))
        y = int(np.round(bbox[1]*scale_y, 4))
        x1 = int(np.round(bbox[2]*(scale_x), 4))
        y1= int(np.round(bbox[3]*scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

# To log a bounding box, you'll need to provide a dictionary with 
# the following keys and values to the boxes keyword argument of wandb.Image.
def wandb_bbox(image, bboxes, true_label, class_id_to_label):
    all_boxes = []
    for bbox in bboxes:
        box_data = {"position": {
                        "minX": bbox[0],
                        "minY": bbox[1],
                        "maxX": bbox[2],
                        "maxY": bbox[3]
                    },
                     "class_id" : int(true_label),
                     "box_caption": class_id_to_label[true_label],
                     "domain" : "pixel"}
        all_boxes.append(box_data)
    

    return wandb.Image(image, boxes={
        "ground_truth": {
            "box_data": all_boxes,
          "class_labels": class_id_to_label
        }
    })