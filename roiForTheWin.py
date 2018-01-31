class RoiFTW(object):

    def __init__(self, value):
        self.value = value
        self.glXyCoords = []
        self.dlXyCoords = []
        self.gdXyCoords = []
        self.ddXyCoords = []
        self.intersectBlue = False
        self.intersectGreen = False
        self.key = 0

    def setKey(self, key):
        self.key = key

    def setValue(self, value):
        self.value = value

    def addGLxyCoords(self, x, y):
        self.glXyCoords.append([x, y])

    def addDLxyCoords(self, x, y):
        self.dlXyCoords.append([x, y])

    def addGDXyCoords(self, x, y):
        self.gdXyCoords.append([x, y])

    def addDDXyCoords(self, x, y):
        self.ddXyCoords.append([x, y])

    def setIntersectsBlue(self, intersect):
        self.intersectBlue = intersect
    
    def setIntersectsGreen(self, intersect):
        self.intersectGreen = intersect
