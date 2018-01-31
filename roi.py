class Roi(object):

    def __init__(self, value, key):
        self.value = value
        self.xyCoords = []
        self.intersectBlue = False
        self.intersectGreen = False
        self.key = key
        
    def getKey(self):
        return self.key

    def getValue(self):
        return self.value

    def getXyCoords(self):
        return self.xyCoords

    def getIntersectsBlue(self):
        return self.intersectBlue

    def getIntersectsGreen(self):
        return self.intersectGreen

    def setValue(self, value):
        self.value = value

    def addXyCoords(self, x, y):
        self.xyCoords.append([x, y])

    def setIntersectsBlue(self, intersect):
        self.intersectBlue = intersect
    
    def setIntersectsGreen(self, intersect):
        self.intersectGreen = intersect
