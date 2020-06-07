from PIL import Image, ImageDraw
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SoftmaxLayer, GaussianLayer, SigmoidLayer, TanhLayer
from math import sin, cos
LAYERS = [4,100,75,50,10,50,75,100,1]
#LAYERS = [4,50,50,50,50,50,50,1]
SIZE = (1000,600)

# We'll do black and white for this one
# so all colors will have the same level per pixel
# Note: Softmax is not good as an output layer

if __name__ == '__main__':
    for i in range(1):
        result = Image.new('RGB',SIZE)
        draw = ImageDraw.Draw(result)
        network = buildNetwork(*LAYERS, hiddenclass=TanhLayer, outclass=SigmoidLayer)
        print('Network created...')
        for x in range(SIZE[0]):
            x1 = float(x)*20/SIZE[0]
            for y in range(SIZE[1]):
                y1 = float(y)*12/SIZE[1]
                z = pow(x1**2 + y1**2, 1/2)
                #z = sin(x+y)
                bias = 10
                output = network.activate([x1,y1,z,bias])
                r,g,b = map(lambda x: int(x*225), list(output)*3)
                draw.point([x,y], fill=(r,g,b))
        result.save(str(i) + 's.png', 'PNG')
        print(str(i) + 's.png saved')
