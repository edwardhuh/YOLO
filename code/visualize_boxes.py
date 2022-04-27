from PIL import Image, ImageDraw

def visualize(filepath, bounding_boxes, found_boxes):
    img = Image.open(filepath)
    Drawer = ImageDraw.Draw(img)
    for box in bounding_boxes:
        box = [i * 256 for i in box]
        box[0] -= box[2] // 2
        box[1] -= box[3] // 2
        box[2] += box[0]
        box[3] += box[1]
        Drawer.rectangle(box, fill=None)
    for box in found_boxes:
        continue
        Drawer.rectangle([i * 256 for i in box], fill=None, outline="red")
    img.show()

if __name__ == "__main__":
    visualize("../data/images/CrowdHuman_train01/Images/resized/273275,cd061000af95f691_256.jpg", [[0.495, 0.49166666666666664, 0.041666666666666664, 0.05625]], [])

