"""
Utility function for visualizing individual bounding boxes
"""
from PIL import Image, ImageDraw


def visualize(filepath, bounding_boxes, found_boxes):
    img = Image.open(filepath)
    Drawer = ImageDraw.Draw(img)
    for box in bounding_boxes:
        box = [i * 416 for i in box]
        box[0] -= box[2] // 2
        box[1] -= box[3] // 2
        box[2] += box[0]
        box[3] += box[1]
        Drawer.rectangle(box, fill=None)
    for box in found_boxes:
        # continue
        Drawer.rectangle([i * 416 for i in box], fill=None, outline="red")
    img.show()
