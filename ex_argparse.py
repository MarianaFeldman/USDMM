import argparse

# Inicializa o parser de argumentos
parser = argparse.ArgumentParser(description="Descrição do seu script")

# Adiciona argumentos específicos com nomes distintos
parametros = ['new_parameter', 'var_type', 'var_name', 'param_a', 'param_b', 'param_c', 'param_d', 'param_e', 'param_f', 'param_g']
for param in parametros:
    parser.add_argument(f'--{param}', type=str, required=True, help=f"Descrição do parâmetro {param}")

# Analisa os argumentos passados
args = parser.parse_args()

# Torna os parâmetros variáveis globais
for param in parametros:
    globals()[param] = getattr(args, param)

# Seu código pode usar as variáveis new_parameter, var_type, var_name, ..., param_g aqui
for param in parametros:
    print(f"O valor de {param} é: {globals()[param]}")

# Exemplo de uso das variáveis em outro ponto do código
def exemplo_de_uso():
    for param in parametros:
        print(f"Usando a variável global {param}: {globals()[param]}")

# Chama a função para verificar o uso
exemplo_de_uso()
'''
python test.py --new_parameter valor0 --var_type valor1 --var_name valor2 --param_a valor3 --param_b valor4 --param_c valor5 --param_d valor6 --param_e valor7 --param_f valor8 --param_g valor9
'''