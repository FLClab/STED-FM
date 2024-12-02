
from PIL import Image, ImageDraw, ImageFont

def generate_image(text):

    # get an image
    base = Image.new("RGB", (224, 224), (34, 34, 34))

    # get a font
    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 120)
    # get a drawing context
    d = ImageDraw.Draw(base)

    # draw text, half opacity
    d.text((112, 112), text, font=fnt, fill=(255, 255, 255), anchor="mm")

    base.save(f"./candidates/{text}.png")

for i in range(5):
    generate_image(str(i))
