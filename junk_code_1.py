import numpy as np


# Criando um array NumPy de 1D
array_1d = np.array([1, 2, 3, 4, 5])
print("Array 1D:")
print(array_1d)

# Operações básicas com o array
print("\nSoma de todos os elementos:", np.sum(array_1d))
print("Média dos elementos:", np.mean(array_1d))
print("Desvio padrão dos elementos:", np.std(array_1d))

# Criando uma matriz 2D (2x3)
matriz_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\nMatriz 2D:")
print(matriz_2d)

# Operações básicas com a matriz
print("\nSoma de todos os elementos da matriz:", np.sum(matriz_2d))
print("Soma ao longo das colunas:", np.sum(matriz_2d, axis=0))
print("Soma ao longo das linhas:", np.sum(matriz_2d, axis=1))

# Operações de álgebra linear: multiplicação de matrizes
matriz_a = np.array([[1, 2], [3, 4]])
matriz_b = np.array([[5, 6], [7, 8]])
resultado_multiplicacao = np.dot(matriz_a, matriz_b)

print("\nMultiplicação de duas matrizes:")
print(resultado_multiplicacao)

# Criando um array de zeros e outro de uns
array_zeros = np.zeros((2, 3))
array_uns = np.ones((3, 2))

print("\nArray de zeros (2x3):")
print(array_zeros)

print("\nArray de uns (3x2):")
print(array_uns)
