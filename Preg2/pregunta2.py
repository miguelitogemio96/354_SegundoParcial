"""
Created on Sun Dec 13 19:13:35 2020

@author: migue
"""
import random


def conversion(num):
    lista = []
    if num == 0:
        resultado = "0"
    else:
        while num >= 1:
            lista.insert(0, num%2)
            num = num // 2
        resultado = "".join(str(i) for i in lista)
    return resultado



def complemento (lista):
    lista2 = []
    for i in range(10):
        while len(lista[i]) < 6:
            lista[i] = "0" + lista[i]
        lista2.append(lista[i])
    return lista2



def cruce (lista):
    a = 0
    ls = []
    aux1 = ""
    aux2 = ""
    aux3 = ""
    aux = ""
    while a < 10:
        aux2 = lista[0][a]
        aux3 = lista[0][a + 1]
        aux1 = aux2[:4] + aux3[4:]
        aux = aux3[:4] + aux2[4:]
        ls.append(aux1)
        ls.append(aux)
        a = a + 2
    return ls



def mutacion (lista):
    aux = ""
    i = 0
    while i < 10:
        aux = lista[0][i]
        if aux[2] == '0':
            lista[0][i] = aux[:2] + '1' + aux[3:]
        else:
            lista[0][i] = aux[:2] + '0' + aux[3:]
        i = i + 1
    return lista



def binario_a_decimal(numero_binario):
    numero_decimal = 0
    for posicion, digito_string in enumerate(numero_binario[::-1]):
        numero_decimal += int(digito_string) * 2 ** posicion
    return numero_decimal


x = []
for i in range(10):
    x.append(random.randint(0, 20))
cont = 0
while(cont <= 3):
    cont+=1
    binX = []
    comp =[]
    cruc = []
    mutac = []
    func = []
    print("GENERACION {}".format(cont))
    #print(cont)
    print("")
    print(x)
    x.sort()
    #print(x)
    x.reverse()
    print(x)
    for i in range(10):
        binX.append(conversion(x[i]))
    for i in range(10):
        #La funcion x*3+x**2+x
        func.append(x[i]*x[i]*x[i] + x[i]*x[i] + x[i])
    print("la poblacion es: \n")
    print(func)
    print("")
    print(binX)
    comp.append(complemento(binX))
    print(comp)
    cruc.append(cruce(comp))
    print(cruc)
    mutac.append(mutacion(cruc))
    print(mutac)
    for i in range(10):
        x[i] = binario_a_decimal(mutac[0][0][i])
    print(x)
   






