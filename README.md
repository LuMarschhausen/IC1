# <h1 align="center"> Respostas e minhas Interpretações acerca dos resultados </h1>

# <h1 align="center"> Perceptron </h1>

- **Questão 1**  
<img src="imagens/1.png">  
Média de iterações para convergir: 10.063  
Média de divergência P[f(x) ≠ g(x)]: 0.109334  
**Resposta (b) aprox 15**   
Este resultado demonstra conceitos importantes de aprendizado de máquina, como a capacidade de um algoritmo de aprender uma função
a partir de exemplos rotulados (aprendizado supervisionado), o processo de ajuste iterativo dos parâmetros do modelo (online) 
e a avaliação de desempenho do modelo em dados não vistos (generalização). Além disso, destaca a importância de avaliar a eficiência e a eficácia dos algoritmos de aprendizado, tanto em termos de número de iterações para convergência quanto em termos de precisão da classificação 
em novos dados.

- **Questão 2**  
Média de divergência P[f(x) ≠ g(x)]: 0.1095244  
**Resposta (c) aprox 0.1**  
Esta questão está diretamente relacionada aos conceitos de generalização e capacidade de aprendizado. A capacidade de um modelo de generalizar para novos dados fora do conjunto de treinamento é muito importante, como vimos em aprendizado de máquina. A divergencia pedida, quantifica o desempenho do modelo em dados não vistos, permitindo avaliar se o Perceptron, apesar de ser um algoritmo simples, consegue aprender uma boa aproximação da função target a partir de um pequeno conjunto de dados.

- **Questão 3**  
Média de iterações para convergir: 118.22  
**Resposta (b) aprox 100**  
Esta questão está relacionada aos conceitos de eficiência do algoritmo e complexidade computacional. A eficiência 
com a qual um algoritmo converge para uma solução é importantíssima, especialmente com conjuntos de dados maiores. O número de iterações para a convergência do PLA reflete a rapidez com que o algoritmo ajusta seus pesos para classificar corretamente todos os pontos de treinamento. 

- **Questão 4**  
Média de divergência P[f(x) ≠ g(x)]: 0.013131900000000002  
**Resposta (b) aprox 0.01**  
Esta questão relaciona-se diretamente com os conceitos de overfitting, underfitting, e capacidade de generalização em aprendizado de máquina. 
Ao aumentar o número de pontos de treinamento, o modelo obtido pelo Perceptron tende a capturar melhor a verdadeira distribuição dos dados, 
reduzindo a divergência entre a função target f e hipótese g. Isso mostra que, com mais dados, o modelo é capaz de generalizar melhor, reduzindo 
o risco de overfitting e underfitting.

- **Questão 5**  
Sim, à medida que o número de pontos de treinamento N aumenta, é esperado que a probabilidade de divergência entre a função target f e a hipótese g encontrada pelo Perceptron, diminua. Isso ocorre porque um conjunto de treinamento maior proporciona mais informações para o algoritmo de aprendizado, resultando em uma hipótese mais precisa e com menor erro de generalização. Porém, o aumento em N também pode levar a um aumento 
no número médio de iterações necessárias para a convergência do Perceptron, devido à complexidade adicional dos dados. Portanto, há um trade-off 
entre a capacidade de generalização do modelo e a eficiência computacional do algoritmo de aprendizado, logo, aumentando a importância de escolher 
um tamanho adequado para o conjunto de treinamento em problemas de aprendizado de máquina.

# <h1 align="center"> Regressão Linear </h1>


- **Questão 1**  
Média de E_in: 0.039180000000000006  
**Resposta (c) aprox 0.01**  
O resultado de E_out médio reflete a eficácia da Regressão Linear em classificar corretamente os pontos de treinamento. Em aprendizado de máquina, 
E_in representa o erro dentro da amostra, indicando quão bem o modelo se ajusta aos dados de treinamento. Um E_in baixo, sugere que o modelo está 
aprendendo bem a relação entre os pontos de entrada e seus rótulos dentro da amostra, enquanto um E_in mais alto mais alto pode indicar underfitting. 
Esta medida é importante para avaliar a performance inicial do modelo antes de testar sua capacidade de generalização em novos dados. Ou seja, E_in 
é um indicador primário da qualidade do aprendizado do modelo no conjunto de treinamento.


- **Questão 2**  
Média de E_out: 0.04808300000000001  
**Resposta (c) aprox 0.01**  
A métrica E_out reflete a capacidade do modelo de generalizar para novos dados que não foram vistos durante o treinamento. Em aprendizado de 
máquina, um modelo bem ajustado deve apresentar um E_out baixo, indicando que ele consegue manter um bom desempenho em dados fora da amostra. 
Comparar E_in com E_out​ ajuda a identificar se o modelo está sofrendo de overfitting, onde ele performa bem nos dados de treinamento (E_in ​
baixo) mas mal nos dados de teste (E_out alto). A proximidade entre E_in e E_out neste experimento sugere que a Regressão Linear conseguiu 
aprender uma hipótese que generaliza bem, demonstrando sua eficácia como técnica de aprendizado de máquina para problemas de classificação linear.

- **Questão 3**  
Média de iterações: 4.136  
**Resposta (a) aprox 1**  
O resultado está relacionado aos conceitos de eficiência do algoritmo e initialization em aprendizado de máquina. Utilizar pesos iniciais obtidos 
por Regressão Linear pode fornecer ao PLA um ponto de partida mais próximo da solução ideal, o que pode reduzir significativamente o número de 
iterações necessárias para convergência. Este experimento demonstra a importância da escolha de uma boa inicialização dos pesos para acelerar a 
convergência dos algoritmos de aprendizado. Em suma, bons métodos de inicialização podem melhorar a eficiência computacional e a velocidade de 
treinamento de modelos de aprendizado de máquina.

- **Questão 4**  
<img src="imagens/4.png"> 
<img src="imagens/4-2.png">   
Média de E_in: 0.10739000000000003  
Média de E_out: 0.03643  
**Resposta (d)**  
Esse experimento demonstra a importância da inicialização dos pesos e do número de iterações no desempenho de algoritmos de aprendizado de máquina. Inicializar os pesos com Regressão Linear fornece ao PLA um ponto de partida mais próximo da solução ideal, melhorando a eficiência e a eficácia do treinamento. O número de iterações também é crucial, pois um maior número de iterações permite ao algoritmo explorar melhor o espaço de hipóteses e ajustar-se de maneira mais precisa aos dados de treinamento. Este experimento destaca a necessidade de um bom equilíbrio entre inicialização adequada e quantidade de iterações para alcançar um desempenho ótimo em termos de erros dentro e fora da amostra (E_in e E_out) refletindo a capacidade do modelo de aprender e generalizar bem a partir dos dados disponíveis.

# <h1 align="center"> Regressão Não-Linear </h1>
- Questão 1  
Média de E_in: 0.504521  
**Resposta (d)**  
O resultado desse experimento mostra os desafios e considerações importantes de aprendizado de máquina, incluindo o manejo de ruído, a importância da complexidade do modelo, e a necessidade de transformações de atributos para melhorar a capacidade de generalização do modelo.

- Questão 2  
Média de E_in: 0.123675  
Vetor de pesos w: [-0.97761588 , 0.08052633 , 0.00368432 , 0.13947378 , 1.58248726 , 1.54154885]  
**Resposta (a)**  
 O resultado deste experimento demonstra como a transformação de atributos pode melhorar a performance de modelos lineares ao capturar relações não-lineares nos dados, um conceito central em aprendizado de máquina para melhorar tanto o ajuste quanto a capacidade de generalização do modelo.  

- Questão 3  
Média de E_out: 0.14297100000000001  
**Resposta (b) aprox 0.1**  
Este resultado demonstra a importância de avaliar a capacidade de generalização do modelo, além de seu desempenho nos dados de treinamento. O uso de transformações adequadas pode melhorar a performance do modelo sem sacrificar sua capacidade de generalizar para novos dados.











