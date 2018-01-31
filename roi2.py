class Roi2(object):

    def __init__(self, value, key):
        self.value = value
        self.gXyCoords = []
        self.dXyCoords = []
        self.intersectBlue = False
        self.intersectGreen = False
        self.key = key
        
    def getKey(self):
        return self.key

    def getValue(self):
        return self.value

    def getGXyCoords(self):
        return self.gXyCoords
     
    def getDXyCoords(self):
        return self.gXyCoords

    def getIntersectsBlue(self):
        return self.intersectBlue

    def getIntersectsGreen(self):
        return self.intersectGreen

    def setValue(self, value):
        self.value = value

    def addGXyCoords(self, x, y):
        self.gXyCoords.append([x, y])

    def addDXyCoords(self, x, y):
        self.dXyCoords.append([x, y])

    def setIntersectsBlue(self, intersect):
        self.intersectBlue = intersect
    
    def setIntersectsGreen(self, intersect):
        self.intersectGreen = intersect
