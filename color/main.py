from PIL import Image, ImageDraw
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SoftmaxLayer, GaussianLayer, SigmoidLayer, TanhLayer
from math import sin, cos
LAYERS = [4,100,75,50,75,100,3]
SIZE = (3000,1800)

# Color for this one

if __name__ == '__main__':
    for i in range(100):
        result = Image.new('RGB',SIZE)
        draw = ImageDraw.Draw(result)
        network = buildNetwork(*LAYERS, hiddenclass=TanhLayer, outclass=SigmoidLayer)
        print('Network created...')
        for x in range(SIZE[0]):
            x1 = float(x)*20/SIZE[0]
            for y in range(SIZE[1]):
                y1 = float(y)*12/SIZE[1]
                z = pow(x1**2 + y1**2, 1/2)
                bias = 10
                rgb = network.activate([x1,y1,z,bias])
                r,g,b = map(lambda x: int(x*255), rgb)
                draw.point([x,y], fill=(r,g,b))
        result.save(str(i) + '.png', 'PNG')
        print(str(i) + '.png saved')
