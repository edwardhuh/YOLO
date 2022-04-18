from PIL import Image, ImageDraw

def visualize(filepath, bounding_boxes):
    img = Image.open(filepath)
    Drawer = ImageDraw.Draw(img)
    for box in bounding_boxes:
        Drawer.rectangle([i * 256 for i in box], fill=None)
    img.show()

if __name__ == "__main__":
    visualize("../data/images/CrowdHuman_train01/Images/resized/273275,cd061000af95f691_256.jpg", [[285 / 600, 223 / 480, (25 + 285) / 600, (27 + 223) / 480], [193 / 600, 135 / 480, 89 / 600, 92 / 480]])

