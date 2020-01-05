#########################################################################################################################
#	Pramodkumar Gupta
# 	Knowledgebase for Python
# 	Log History: 1. Map Function
#				 2. Filter Function
#
#########################################################################################################################



#----------------------------------- Map Function -------------------------------------------------------------

# map function is used to perform operation over many values placed inside list or say iterator.

lst=[1,2,3,4,5,6,7,8,9,0]

list(map(lambda num: num % 2 == 0, lst))

# o/p: [False, True,False, True,False, True,False, True,False, True]


#----------------------------------- Filter Function ----------------------------------------------------------

# Filter  function is used to perform operation over many values placed inside list or say iterator and it will only return true condition data.
lst=[1,2,3,4,5,6,7,8,9,0]

list(filter(lambda num: num % 2 == 0, lst))

# o/p: [2,4,6,8,0]
