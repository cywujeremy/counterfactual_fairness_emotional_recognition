male_total = 5239
female_total = 4800
ratio = male_total/female_total

data = [[1029, 820], [975, 733], [443, 698]]
diff = []
for m, f in data:
    adjusted_f = f * ratio
    adjusted_total = adjusted_f + m
    dif_ratio = (m - adjusted_f)/adjusted_total
    diff.append(dif_ratio)

print(diff)