import string
import codecs
from unidecode import unidecode


with codecs.open("en.txt", 'r', 'utf-8') as inf:
	mod_lines = ['']

	#sp_file = open("spanish2.txt", "r")
	#sp_lines = sp_file.readlines();

	for line in inf:
		sOut = unidecode(line.strip())
		sOut = sOut+' endsen\r\n'
		mod_lines.append(sOut)

new_file = open("en.txt", "w")
new_file.writelines(mod_lines)
new_file.close

