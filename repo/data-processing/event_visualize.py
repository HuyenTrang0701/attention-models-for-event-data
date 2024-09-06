from pathlib import Path
import numpy as np
import cv2
import argparse
import sys

# Add the parent directory of 'dsec_det' to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsec_det.dataset import DSECDet
from dsec_det.visualize import render_events_on_image
from dsec_det.label import COLORS, CLASSES


def render_object_detections_on_image_with_prob(img, tracks, **kwargs):
    return _draw_bbox_on_img_with_prob(img, tracks['x'], tracks['y'], tracks['w'], tracks['h'],
                                       tracks['class_id'], tracks['class_confidence'], **kwargs)

def _draw_bbox_on_img_with_prob(img, x, y, w, h, labels, scores=None, conf=0.5, label="", scale=1, linewidth=2, show_conf=True):
    for i in range(len(x)):
        if scores is not None and scores[i] < conf:
            continue

        x0 = int(scale * (x[i]))
        y0 = int(scale * (y[i]))
        x1 = int(scale * (x[i] + w[i]))
        y1 = int(scale * (y[i] + h[i]))
        cls_id = int(labels[i])

        color = (COLORS[cls_id] * 255).astype(np.uint8).tolist()

        text = f"{label}-{CLASSES[cls_id]}"

        if scores is not None and show_conf:
            text += f":{scores[i] * 100: .1f}%"

        txt_color = (0, 0, 0) if np.mean(COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, linewidth)

        txt_bk_color = (COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_height = int(1.5 * txt_size[1])
        cv2.rectangle(
            img,
            (x0, y0 - txt_height),
            (x0 + txt_size[0] + 1, y0 + 1),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1] - txt_height), font, 0.4, txt_color, thickness=1)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualize an example.""")
    parser.add_argument("--dsec_merged", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    assert args.split in ['train', 'test']
    assert args.dsec_merged.exists() and args.dsec_merged.is_dir()

    dataset = DSECDet(args.dsec_merged, split=args.split, sync="back", debug=True)

    while True:
        index = np.random.randint(0, len(dataset))
        #print("=>>>>>> index ", index)
        try:
            output = dataset[index]
            #print("=>>>>>> output ", output)
            
            # Create a blank image for visualization
            blank_image = np.zeros((480, 640, 3), np.uint8)
            blank_image.fill(255)  # White background for better visibility of events

            # Render events on the blank image
            blank_image = render_events_on_image(blank_image, x=output['events']['x'], y=output['events']['y'], p=output['events']['p'])
            
            # Render bounding boxes with probabilities on the blank image
            blank_image = render_object_detections_on_image_with_prob(blank_image, output['tracks'])
            
            cv2.imshow("Visualization", blank_image)
        except Exception as e:
            print("Error:", e)
        cv2.waitKey(0)
