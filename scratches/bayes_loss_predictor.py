
class LossPredictor:
    def __init__(self):
        self.losses = []
        self.prev = None
        self.ups = 0
        self.downs = 0

    def update(self, loss):
        self.losses.append(loss)
        if self.prev:
            if loss < self.prev:
                self.downs += 1
            else:
                self.ups += 1
        self.prev = loss

    def ratio(self):
        total = self.ups + self.downs
        if total > 0:
            return self.downs / total
        else:
            return self.downs

    def past_average(self, count):
        sum = 0
        if count > len(self.losses):
            count = len(self.losses)
        for loss in self.losses[-count:]:
            sum = sum +  loss
        return sum / count

if __name__ == '__main__':

    losses = [99,94,23,22]

    nlp = LossPredictor()
    for loss in losses:
        nlp.update(loss)

    print(nlp.ratio())
    print(nlp.past_average(2))





