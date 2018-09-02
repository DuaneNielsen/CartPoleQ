import pickle
PIK = "pickle.dat"

data = ["A", "b", "C", "d"]

with open(PIK, "wb") as f:
    pickle.dump(data, f)

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

for item in loadall(PIK):
    print(item)

with open(PIK, "wb") as f:
    pickle.dump(len(data), f)
    for value in data:
        pickle.dump(value, f)
data2 = []
with open(PIK, "rb") as f:
    for _ in range(pickle.load(f)):
        data2.append(pickle.load(f))

with open(PIK, 'wb') as f:
    for value in data:
        pickle.dump(value, f)

with open(PIK, "rb") as f:
    data3 = pickle.load(f)
    data4 = pickle.load(f)

print(data3)
print(data4)
