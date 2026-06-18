def int_to_binary_list(n, bits = 8):
    twos_complement = n & (2**bits - 1)
    return [int(bit) for bit in format(twos_complement, f'0{bits}b')]

idx = "d0"
out = "c0"

#1 -> right; -1 -> left; 0 -> stay
fp = open("dt/"+idx+".txt")

dt = []
bias_x = 2
bias_y = 1
for line in fp.readlines():
	text = line.strip().split("\t")
	ball_x, ball_y, paddle_x, paddle_y, act = text
	tmp = int_to_binary_list(int(ball_x) + 2)
	tmp.extend(int_to_binary_list(int(ball_y) + 2))
	tmp.extend(int_to_binary_list(int(paddle_x) + 2))
	dt.append([tmp, int_to_binary_list(int(act) + 1, 2)])

fp.close()


#0,0 -> left; 1,0 -> right; 0,1 -> stay
fp = open("dt/"+idx+"_d.txt", 'w')
for d in dt:
	fp.write(','.join(str(t) for t in d[0]) + '\t' + ','.join(str(t) for t in d[1]) + '\n')

fp.close()

fp = open("dt/"+idx+"_d.txt")

new_d = []
new_y = []
#0,1,0 -> left; 1,0,0 -> right; 0,0,1 -> stay
for line in fp.readlines():
	x, y = line.strip().split("\t")
	if y == "1,0":
		y_new = "1,0,0"
	elif y == '0,0':
		y_new = "0,1,0"
	else:
		y_new = "0,0,1"
	new_d.append(x)
	new_y.append(str(y_new))
fp.close()

fp = open("dt/"+out+".txt", 'w')
for x, y in zip(new_d, new_y):
	fp.write(x+"\t"+y+"\n")
fp.close()



