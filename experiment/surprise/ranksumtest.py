setA = [10, 22, 42, 59, 61, 63, 65, 83, 85, 90, 93]
setB = [36, 53, 54, 56, 69, 84, 88]

setA = sorted(setA)
setB = sorted(setB)

print(setA)

setAsize = len(setA)
setBsize = len(setB)

setAmarked = []
setBmarked = []

for datapoint in setA:
    setAmarked.append([datapoint, 'A'])

for datapoint in setB:
    setBmarked.append([datapoint, 'B'])

combinedSet = sorted(setAmarked + setBmarked)

# print combinedSet

setAranks = []
setBranks = []

for rank, data in enumerate(combinedSet):
    print(rank)
    if data[1] == 'A':
        setAranks.append(rank + 1)

    elif data[1] == 'B':
        setBranks.append(rank + 1)


setArankTotal = sum(setAranks)
setBrankTotal = sum(setBranks)

if setAsize <= setBsize:
    rankSum = setArankTotal
    m = setAsize
    n = setBsize
else:
    rankSum = setBrankTotal
    n = setAsize
    m = setBsize


halfMsum = 0.5 * m * (m + n + 1)
twelthMNsum = halfMsum * n / 6

zNumerator = rankSum - halfMsum

if zNumerator >= 0:
    zNumerator -= 0.5
else:
    zNumerator += 0.5


zDenominator = ((1.0 / 6) * halfMsum * n) ** 0.5

z = zNumerator / zDenominator

# print "A raw data : ", setA
# print "B raw data : ", setB

# print "Set A ranks : ", setAranks
# print "Set B ranks : ", setBranks

# print "Set A size : ", setAsize
# print "Set B size : ", setBsize

# print "Set A rank total : ", setArankTotal
# print "Set B rank total : ", setBrankTotal

# print "Rank Sum (Rm) : ", rankSum

# print "halfMsum : ", halfMsum
# print "twelthMNsum : ", twelthMNsum
# print "Z numerator : ", zNumerator
# print "Z denominator : ", zDenominator
# print "Z : ", z
