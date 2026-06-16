# Briscola RL

## Specifica progettuale da validare

## 1. Idea del progetto

Il progetto consiste nella costruzione di un agente di Reinforcement Learning
capace di giocare una partita completa di **Briscola a due giocatori**.

L'agente deve imparare principi generali di buona strategia attraverso il
gioco, senza ricevere una lista esplicita di regole strategiche come:

- conserva sempre le briscole;
- gioca un carico soltanto in determinate situazioni;
- usa una briscola solo quando nella presa sono presenti molti punti.

L'agente conosce le regole del gioco, osserva la propria mano e tutta
l'informazione pubblica disponibile, e sceglie quale carta giocare.

L'addestramento avviene tramite **self-play**. Una policy learner gioca contro
copie congelate di versioni precedenti della stessa policy. Le copie storiche
sono conservate in un pool, così che il learner non si specializzi soltanto
nel battere la propria versione immediatamente precedente.

L'obiettivo non è dimostrare di avere risolto matematicamente la Briscola, né
garantire la vittoria contro qualunque avversario. L'obiettivo è ottenere una
policy generale e robusta che:

- giochi legalmente una partita completa;
- utilizzi la memoria delle carte già giocate;
- tenga conto del punteggio e della fase della partita;
- migliori rispetto alla policy iniziale;
- generalizzi contro avversari non usati per gli aggiornamenti;
- possa sostenere partite sensate contro giocatori umani.

## 2. Domande di ricerca

Il progetto risponde principalmente alle seguenti domande.

1. Una policy addestrata esclusivamente tramite self-play riesce a imparare
   strategie efficaci nella Briscola a due?
2. Quanto contribuisce la memoria delle carte già giocate rispetto a una
   policy che osserva soltanto la mano e la presa corrente?
3. Un pool di snapshot storiche produce una policy più robusta rispetto al
   self-play contro la sola versione più recente?

La terza domanda è centrale. Il pool non contiene "personalità" artificiali o
modelli psicologici dell'avversario. Contiene soltanto versioni precedenti
della stessa policy.

## 3. Regole e perimetro del gioco

Si considera esclusivamente la Briscola standard a due giocatori.

Assunzioni:

- mazzo italiano da 40 carte;
- quattro semi;
- tre carte iniziali per giocatore;
- una carta scoperta determina il seme di briscola;
- nessun obbligo di rispondere al seme;
- il vincitore della presa pesca per primo e apre la presa successiva;
- una smazzata termina quando tutte le 40 carte sono state giocate;
- ogni giocatore gioca esattamente 20 carte;
- la somma dei punti disponibili è 120.

Ordine di presa, dal più forte al più debole:

```text
Asso, Tre, Re, Cavallo, Fante, Sette, Sei, Cinque, Quattro, Due
```

Valori:

| Carta | Punti |
|---|---:|
| Asso | 11 |
| Tre | 10 |
| Re | 4 |
| Cavallo | 3 |
| Fante | 2 |
| Altre | 0 |

Il progetto non comprende:

- Briscola a squadre;
- Briscola chiamata;
- segnali tra compagni;
- più mani per incontro;
- scommesse o premi monetari;
- modellazione dello stile psicologico del giocatore umano.

## 4. Natura matematica del problema

### 4.1 Gioco completo

La Briscola a due non è, nella sua forma completa, un normale MDP
single-agent. È un **gioco stocastico a somma zero e informazione imperfetta**.

Indichiamo il gioco con:

\[
\mathcal{G} =
\left(
\mathcal{I},
\mathcal{S},
\{\mathcal{A}_i\},
P,
\{\mathcal{O}_i\},
Z,
\{R_i\},
\gamma
\right)
\]

dove:

- \(\mathcal{I}=\{1,2\}\) è l'insieme dei giocatori;
- \(\mathcal{S}\) è lo spazio degli stati reali;
- \(\mathcal{A}_i\) è lo spazio delle azioni del giocatore \(i\);
- \(P\) descrive giocate, pescate e avanzamento della partita;
- \(\mathcal{O}_i\) è lo spazio delle osservazioni del giocatore \(i\);
- \(Z\) è la funzione di osservazione;
- \(R_1=-R_2\) rende il gioco a somma zero;
- \(\gamma=1\), perché la partita è episodica e finita.

### 4.2 Stato reale

Lo stato reale dell'ambiente al tempo \(t\) contiene:

\[
S_t =
\left(
H_t^1,
H_t^2,
D_t,
\tau,
B_t,
C_t,
P_t^1,
P_t^2,
\ell_t,
i_t
\right)
\]

dove:

- \(H_t^i\) è la mano del giocatore \(i\);
- \(D_t\) è il mazzo residuo ordinato;
- \(\tau\) è il seme di briscola;
- \(B_t\) è la carta di briscola scoperta, finché rimane visibile;
- \(C_t\) è la presa corrente, vuota o contenente la prima carta;
- \(P_t^i\) è il punteggio acquisito dal giocatore \(i\);
- \(\ell_t\) indica chi conduce la presa;
- \(i_t\) indica chi deve agire.

Conoscendo \(S_t\), l'azione corrente e le regole, la distribuzione del futuro
non dipende dalla storia precedente. Lo stato reale soddisfa quindi la
proprietà di Markov.

### 4.3 Osservazione del giocatore

Il giocatore non osserva:

- la mano avversaria;
- l'ordine delle carte nel mazzo;
- le carte che l'avversario pescherà.

Osserva invece:

- la propria mano;
- il seme e la carta di briscola scoperta;
- la carta eventualmente giocata nella presa corrente;
- tutte le carte già giocate;
- chi ha giocato ogni carta;
- chi ha vinto ogni presa;
- i punti acquisiti;
- il numero di carte ancora nel mazzo;
- chi conduce la presa.

L'osservazione corrente non coincide con lo stato reale:

\[
O_t^i \neq S_t
\]

Il problema presenta quindi informazione parziale.

## 5. Formulazione usata per l'apprendimento

Durante un batch di addestramento, la policy avversaria è congelata. Dal punto
di vista del learner, l'avversario e le pescate casuali fanno parte
dell'ambiente. Il problema indotto può essere trattato come un POMDP
single-agent:

\[
\mathcal{M}_{\sigma} =
\left(
\mathcal{S},
\mathcal{A},
P_{\sigma},
\mathcal{O},
Z,
R,
\gamma
\right)
\]

dove \(\sigma\) è la policy avversaria fissata per quell'episodio.

Quando viene cambiata la snapshot avversaria, cambia anche il POMDP indotto.
Per mantenere il problema stazionario durante gli aggiornamenti:

- l'avversario viene campionato prima dell'episodio;
- i suoi parametri rimangono congelati per l'intero batch;
- vengono aggiornati soltanto i parametri del learner.

Il self-play complessivo è quindi una successione controllata di problemi
contro avversari congelati, non un singolo MDP stazionario globale.

## 6. Memoria e information state

La policy non riceve le carte nascoste. Riceve una memoria costruita
esclusivamente con informazione legalmente disponibile:

\[
M_t =
\left(
H_t,
\tau,
B_t,
C_t,
U_t,
P_t^{L},
P_t^{O},
N_t,
\ell_t,
\phi_t
\right)
\]

dove:

- \(H_t\) è la mano corrente del learner;
- \(\tau\) è il seme di briscola;
- \(B_t\) è la carta di briscola scoperta, finché non viene pescata;
- \(C_t\) è la presa corrente;
- \(U_t\) rappresenta la storia pubblica delle carte giocate;
- \(P_t^{L}\) e \(P_t^{O}\) sono i punti dei due giocatori;
- \(N_t\) è il numero di carte ancora da pescare;
- \(\ell_t\) indica chi conduce la presa;
- \(\phi_t\) rappresenta la fase della partita.

### 6.1 Storia completa e rappresentazione compressa

Dal punto di vista teorico, la storia completa di azioni e osservazioni
contiene tutta l'informazione disponibile al giocatore:

\[
M_t =
\left(
O_0,A_0,O_1,A_1,\ldots,O_t
\right)
\]

Il progetto sceglie deliberatamente di non fornire questa sequenza completa
alla policy. La policy non usa l'ordine completo delle prese precedenti e non
prova a inferire lo stile o i parametri della policy avversaria. Utilizza:

- la propria mano;
- la carta di briscola scoperta, finché visibile;
- la presa corrente;
- l'insieme delle carte già osservate;
- i punteggi;
- il numero di carte ancora da pescare;
- chi conduce la presa.

Due storie che producono lo stesso riepilogo sono quindi trattate nello
stesso modo. L'incertezza residua dipende soprattutto dalla mano avversaria e
dall'ordine nascosto del mazzo, non dalla necessità di ricostruire
letteralmente tutte le prese precedenti. La policy impara l'azione con ritorno
atteso migliore condizionatamente a questo riepilogo:

\[
a^*(M_t)
=
\arg\max_a
\mathbb{E}[G_t\mid M_t,a]
\]

La rappresentazione resta un information state approssimato del POMDP:
configurazioni nascoste differenti possono corrispondere allo stesso
\(M_t\). REINFORCE può comunque ottimizzare una policy dipendente da
\(M_t\); il limite è che nessuna scelta dei parametri può recuperare
informazioni che non sono presenti nell'input.

Questa scelta riguarda soltanto l'input della policy. Il trainer conserva
temporaneamente la traiettoria:

\[
(M_t,A_t,R_{t+1},\log\pi_\theta(A_t\mid M_t))
\]

per calcolare il reward-to-go e l'aggiornamento REINFORCE. La traiettoria di
training non viene fornita integralmente alla policy per decidere l'azione
successiva.

### 6.2 Divieto di information leakage

La funzione che costruisce le feature del learner non deve accedere:

- alla mano avversaria;
- all'ordine del mazzo;
- alle pescate future;
- ai parametri interni usati dall'avversario per decidere.

L'ambiente può contenere queste informazioni per applicare le transizioni, ma
non deve esporle alla policy.

## 7. Azioni

L'azione del giocatore consiste nel selezionare una carta dalla propria mano:

\[
A_t \in H_t
\]

Il numero di azioni legali è compreso tra uno e tre.

L'implementazione usa un action mask:

\[
\pi_\theta(a\mid M_t)=0
\qquad
\text{se }a\notin H_t
\]

La policy non deve mai proporre una carta che non appartiene alla mano.

La stessa policy parametrica viene usata in entrambi i posti. Ogni giocatore
riceve però una rappresentazione canonica costruita dal proprio punto di
vista: `self` indica sempre il giocatore controllato dalla policy e
`opponent` indica sempre l'altro giocatore. In questo modo non servono due
policy diverse per i due posti.

## 8. Transizioni

### 8.1 Prima carta della presa

Se il learner conduce, la carta selezionata viene inserita nella presa
corrente. L'avversario risponde secondo la policy congelata.

Se l'avversario conduce, la sua carta entra nell'osservazione del learner,
che sceglie la risposta.

### 8.2 Risoluzione della presa

Date le due carte \(c_1\) e \(c_2\):

1. se una sola carta è di briscola, vince la briscola;
2. se entrambe sono di briscola, vince quella di rango superiore;
3. se nessuna è di briscola e hanno lo stesso seme, vince quella di rango
   superiore;
4. se hanno semi diversi e nessuna è di briscola, vince la prima carta.

Il vincitore:

- acquisisce i punti della presa;
- pesca per primo;
- conduce la presa successiva.

### 8.3 Termine dell'episodio

L'episodio termina quando entrambe le mani sono vuote e tutte le 40 carte
sono state giocate.

Devono valere gli invarianti:

\[
P_T^1+P_T^2=120
\]

e:

\[
|G_1|=|G_2|=20
\]

dove \(G_i\) è l'insieme delle carte giocate dal giocatore \(i\).

## 9. Reward e obiettivo

### 9.1 Obiettivi possibili

Una prima possibilità è ottimizzare la differenza finale di punteggio:

\[
G_0=P_T^{L}-P_T^{O}
\]

Questo è un payoff a somma zero:

\[
G_0^{O}=-G_0^{L}
\]

In questo caso, l'obiettivo della policy contro una distribuzione di
avversari \(\rho\) è:

\[
J(\theta;\rho)
=
\mathbb{E}_{\sigma\sim\rho}
\left[
P_T^{L}-P_T^{O}
\right]
\]

### 9.2 Reward per presa

Lo stesso ritorno può essere distribuito senza alterare l'obiettivo della
differenza di punteggio. Per ogni presa:

\[
R_{t+1}=
\begin{cases}
v(c_1)+v(c_2), & \text{se la presa è vinta dal learner}\\
-\left(v(c_1)+v(c_2)\right), & \text{altrimenti}
\end{cases}
\]

Con \(\gamma=1\):

\[
\sum_t R_{t+1}=P_T^{L}-P_T^{O}
\]

Questa reward non prescrive come giocare e rappresenta direttamente la
differenza di punteggio. Non coincide però necessariamente con l'obiettivo
di massimizzare la probabilità di vittoria.

### 9.3 Differenza rispetto al win rate

Massimizzare la differenza attesa di punti non è matematicamente identico a
massimizzare:

\[
\Pr(P_T^L>60)
\]

Una policy che massimizza il margine atteso potrebbe preferire una strategia
più rischiosa, capace di ottenere grandi vittorie ma anche sconfitte più
frequenti, rispetto a una strategia che vince più spesso con margine ridotto.

Per esempio:

- la strategia A vince sempre \(61\)-\(59\);
- la strategia B vince \(100\)-\(20\) nel \(60\%\) dei casi e perde
  \(40\)-\(80\) nel restante \(40\%\).

La strategia A ha win rate \(100\%\) e margine medio \(+2\). La strategia B
ha win rate \(60\%\) e margine medio \(+32\). Una reward basata soltanto sul
margine preferirebbe quindi la strategia B.

Se l'obiettivo prioritario è vincere la partita, il reward terminale naturale
è:

\[
R_T^{\mathrm{win}}=
\begin{cases}
+1, & P_T^L>60\\
0, & P_T^L=60\\
-1, & P_T^L<60
\end{cases}
\]

Questo reward è più aderente all'esito della partita, ma è anche più sparso:
il segnale viene osservato soltanto alla fine e può aumentare la varianza
degli aggiornamenti REINFORCE.

Una possibilità intermedia è usare un reward terminale combinato:

\[
R_T^{\mathrm{combined}}
=
\operatorname{sign}(P_T^L-P_T^O)
+
\lambda\frac{P_T^L-P_T^O}{120}
\]

con \(0<\lambda<1\). Il termine di segno rende prioritario l'esito della
partita, mentre il margine normalizzato distingue risultati con lo stesso
esito e fornisce un segnale secondario sulla qualità del risultato.

Per questo motivo:

- la scelta tra reward sui punti, reward di vittoria e reward combinato deve
  essere dichiarata esplicitamente prima degli esperimenti;
- win rate, draw rate e loss rate sono metriche di valutazione separate;
- una possibile ablation confronta le tre formulazioni;
- la differenza di punti rimane comunque una metrica diagnostica, anche se
  non viene scelta come reward principale.

### 9.4 Nessuna penalità per l'uso delle briscole

Non viene introdotta una reward del tipo:

\[
-\lambda\mathbf{1}
\{\text{è stata giocata una briscola}\}
\]

Usare una briscola può essere corretto o scorretto a seconda dello stato.
Penalizzarla direttamente inserirebbe nella reward una strategia decisa dagli
autori.

"La carta è una briscola" è una feature disponibile alla policy. Sarà
l'apprendimento a determinarne il valore nelle diverse situazioni.

## 10. Rappresentazione della policy

Lo spazio delle storie possibili è troppo grande per una tabella
\(Q(M,a)\). Si usa quindi una policy parametrica con feature interpretabili.

Per ogni carta legale \(a\), si costruisce:

\[
\phi(M_t,a)\in\mathbb{R}^d
\]

La preferenza per l'azione è:

\[
h_\theta(M_t,a)=\theta^\top\phi(M_t,a)
\]

La policy softmax sulle sole azioni legali è:

\[
\pi_\theta(a\mid M_t)=
\frac{\exp(h_\theta(M_t,a))}
{\sum_{a'\in H_t}\exp(h_\theta(M_t,a'))}
\]

La policy è lineare nei parametri, ma può usare feature non lineari e
interazioni costruite esplicitamente.

Quando possibile, le feature dei semi sono espresse relativamente alla
situazione di gioco, per esempio:

- è il seme di briscola;
- è il seme della prima carta della presa;
- è un altro seme.

Questo evita di far apprendere separatamente strategie equivalenti solo
perché coppe, denari, bastoni e spade hanno identificatori differenti.

### 10.1 Feature di base

Le feature possono includere:

- valore in punti della carta;
- rango della carta;
- seme della carta;
- indicatore di briscola;
- gioco da primo o da secondo;
- capacità di vincere la presa corrente;
- punti presenti nella presa corrente;
- carta giocata dall'avversario;
- numero di briscole già uscite;
- numero di briscole non ancora osservate;
- numero di assi e tre non ancora osservati;
- carte superiori dello stesso seme non ancora osservate;
- carte superiori di briscola non ancora osservate;
- punti del learner;
- punti dell'avversario;
- vantaggio corrente;
- numero di carte ancora nel mazzo;
- fase iniziale, centrale o finale;
- indicatore di mazzo esaurito.

### 10.2 Interazioni necessarie

Una policy puramente additiva può non rappresentare decisioni come:

> giocare una briscola è utile solo quando si gioca per secondi e la presa
> contiene molti punti.

Si introducono quindi feature di interazione, per esempio:

- `briscola × gioca_secondo`;
- `briscola × punti_nella_presa`;
- `vince_presa × punti_nella_presa`;
- `carta_alta × fase_finale`;
- `vantaggio × fase_finale`;
- `briscola_alta × briscole_superiori_residue`.

La policy rimane lineare rispetto a \(\theta\), coerentemente con i metodi
lineari trattati nella dispensa.

## 11. Algoritmo di apprendimento

### 11.1 REINFORCE

L'algoritmo principale è REINFORCE, trattato nel capitolo 9 della dispensa.

Per ogni decisione del learner:

\[
\theta
\leftarrow
\theta+
\alpha G_t
\nabla_\theta
\log\pi_\theta(A_t\mid M_t)
\]

dove:

\[
G_t=\sum_{k=t}^{T-1}R_{k+1}
\]

è il reward-to-go dalla decisione corrente alla fine della partita.

L'aggiornamento aumenta la probabilità delle azioni associate a ritorni
positivi e riduce quella delle azioni associate a ritorni negativi.

### 11.2 Riduzione della varianza

REINFORCE può avere varianza elevata. È possibile sottrarre una baseline
indipendente dall'azione:

\[
\theta
\leftarrow
\theta+
\alpha(G_t-b_t)
\nabla_\theta
\log\pi_\theta(A_t\mid M_t)
\]

La versione minima usa:

- una media mobile dei ritorni passati; oppure
- una baseline leave-one-out calcolata sul batch.

Non si introduce un critic neurale e non si trasforma il progetto in un
actor-critic.

### 11.3 Esplorazione

La policy softmax è stocastica e fornisce esplorazione direttamente.

Durante il training non si usa sempre l'azione più probabile. Durante la
valutazione si possono confrontare:

- policy stocastica;
- policy greedy:

\[
a_t=\arg\max_{a\in H_t}\pi_\theta(a\mid M_t)
\]

## 12. Self-play

### 12.1 Perché non aggiornare entrambi i giocatori

Se la stessa policy con gli stessi parametri controlla entrambi i giocatori e
si usano entrambe le traiettorie per aggiornare contemporaneamente gli stessi
parametri, il segnale zero-sum può cancellarsi in aspettativa.

In un gioco simmetrico:

\[
J(\theta,\theta)=0
\]

Per questo si distinguono:

- una policy learner \(\pi_\theta\), aggiornata;
- una policy opponent \(\pi_{\theta^-}\), congelata.

Le due policy possono inizialmente avere parametri identici, ma durante un
batch soltanto \(\theta\) viene modificato.

### 12.2 Avvio dal primo episodio

Il self-play può iniziare dalla prima partita.

Procedura iniziale:

1. inizializzare casualmente \(\theta_0\);
2. creare una copia congelata \(\theta_0^-\);
3. inserire \(\theta_0^-\) nel pool;
4. far giocare il learner contro la copia congelata;
5. aggiornare soltanto il learner.

Non è necessario pre-addestrare l'agente contro policy aggressive,
conservative o scritte manualmente.

## 13. Pool degli avversari

Indichiamo il pool al ciclo \(k\) con:

\[
\mathcal{P}_k=
\{\pi_{\theta^{(0)}},\pi_{\theta^{(1)}},\ldots,
\pi_{\theta^{(m)}}\}
\]

Ogni elemento è una snapshot congelata della stessa famiglia di policy.

### 13.1 Procedura

Per ogni ciclo di training:

1. campionare \(\pi_{\theta^-}\) dal pool;
2. congelare l'avversario per tutto il batch;
3. assegnare casualmente il learner al posto 1 o al posto 2;
4. generare un batch di partite;
5. aggiornare soltanto \(\theta\);
6. ripetere;
7. ogni \(K\) aggiornamenti, valutare il learner;
8. salvare una nuova snapshot nel pool.

### 13.2 Campionamento

La versione iniziale usa campionamento uniforme:

\[
\pi_{\theta^-}\sim
\operatorname{Uniform}(\mathcal{P}_k)
\]

Questo garantisce che il learner affronti sia versioni recenti sia versioni
più vecchie.

Se il pool diventa troppo grande, si conserva un massimo di \(M\) snapshot:

- policy iniziale;
- alcune policy intermedie distribuite nel tempo;
- versioni più recenti;
- migliore versione secondo la valutazione corrente.

La regola di selezione deve essere fissata prima degli esperimenti principali.

### 13.3 Perché il pool è necessario

Allenarsi soltanto contro l'ultima versione può produrre:

- oscillazioni tra strategie;
- specializzazione contro l'avversario corrente;
- dimenticanza di strategie precedentemente efficaci;
- falsa impressione di progresso.

Il pool non garantisce la convergenza a un equilibrio di Nash. È una tecnica
empirica per stabilizzare il self-play e aumentare la diversità degli
avversari incontrati.

## 14. Baseline

Le baseline non sono necessarie per iniziare il training, ma sono necessarie
per capire se l'agente ha imparato qualcosa.

### 14.1 Random

Sceglie uniformemente una carta dalla mano:

\[
\pi_{\text{random}}(a\mid H)=\frac{1}{|H|}
\]

### 14.2 Greedy di presa

Per ogni carta valuta soltanto il risultato immediato della presa:

- prova a vincere i punti presenti;
- non considera conseguenze future;
- non mantiene un modello delle carte residue.

Questa baseline serve a distinguere una policy sequenziale da una strategia
puramente miope.

### 14.3 Euristica semplice

Una singola euristica scritta prima di osservare i risultati può essere usata
come riferimento esterno. Non viene usata per addestrare il learner.

Deve restare semplice e completamente documentata, evitando di trasformare il
progetto in una competizione fra regole manuali.

### 14.4 Snapshot storiche

La policy finale viene confrontata con:

- policy iniziale;
- snapshot intermedie;
- snapshot migliori secondo checkpoint precedenti.

## 15. Disegno sperimentale

### 15.1 Esperimento principale

Confrontare:

1. self-play contro la sola snapshot più recente;
2. self-play contro un pool uniforme di snapshot storiche.

Ipotesi:

> Il pool storico produce una policy con prestazioni più stabili e migliori
> contro una popolazione di test eterogenea.

### 15.2 Ablation sulla memoria

Confrontare:

1. policy senza memoria delle carte uscite;
2. policy con vettore delle carte uscite e informazioni sulle prese;
3. eventuale policy con storia pubblica più ricca.

Ipotesi:

> La memoria fornisce un vantaggio crescente quando il mazzo si riduce e
> l'informazione sulle carte residue diventa più precisa.

### 15.3 Ablation sulla reward

Solo se il tempo lo consente, confrontare:

1. differenza di punti distribuita sulle prese;
2. reward terminale di vittoria:

\[
R_T\in\{-1,0,+1\}
\]

Questo esperimento serve a distinguere:

- qualità dell'obiettivo;
- densità del segnale;
- varianza dell'apprendimento.

Non è necessario per la prima versione completa del progetto.

## 16. Protocollo di valutazione

### 16.1 Separazione training-test

Le valutazioni ufficiali usano:

- seed non usati per selezionare checkpoint;
- avversari congelati;
- nessun aggiornamento dei parametri;
- un numero prefissato di partite.

### 16.2 Controllo della casualità

Per ridurre l'effetto della fortuna:

- si usano coppie di partite con lo stesso seed;
- si scambiano i posti delle due policy;
- si alterna il giocatore iniziale;
- si riportano medie e intervalli di confidenza.

### 16.3 Metriche

Metriche principali:

- differenza media di punti:

\[
\mathbb{E}[P_T^L-P_T^O]
\]

- win rate;
- draw rate;
- loss rate;
- worst-case score contro il pool di test;
- stabilità tra seed differenti.

Metriche diagnostiche:

- frequenza di utilizzo delle briscole;
- punti medi conquistati quando viene usata una briscola;
- frequenza di vittoria delle prese;
- valore medio delle carte sacrificate;
- comportamento nella fase finale;
- prestazione condizionata al vantaggio o svantaggio corrente.

Le metriche diagnostiche descrivono la policy. Non fanno parte della reward.

### 16.4 Matrice degli scontri

Si costruisce una matrice:

\[
W_{ij}=
\text{risultato medio della policy }i
\text{ contro la policy }j
\]

La matrice permette di individuare:

- miglioramento reale;
- cicli fra policy;
- regressioni;
- specializzazioni contro singoli checkpoint.

## 17. Valutazione contro giocatori umani

Le partite contro i membri del gruppo o altri giocatori sono una
dimostrazione qualitativa, non la prova scientifica principale.

Il test umano deve:

- usare la stessa interfaccia e le stesse regole dell'ambiente;
- impedire accesso alle carte nascoste;
- registrare seed, carte e decisioni;
- mostrare a fine partita le probabilità assegnate alle azioni;
- non aggiornare la policy durante l'incontro.

Una piccola quantità di partite umane non consente conclusioni statistiche
forti. Può però mostrare se l'agente:

- gioca senza errori;
- compie scelte plausibili;
- sfrutta la memoria delle carte;
- è sufficientemente competitivo da sostenere una partita reale.

## 18. Collegamento con la dispensa

| Capitolo | Collegamento |
|---|---|
| 1. Introduzione | agente, ambiente, reward, esplorazione |
| 2. MDP/POMDP | stato reale, osservazioni, memoria, information state |
| 4. Monte Carlo | episodi completi e ritorni campionati |
| 8. Approssimazione | policy basata su feature e generalizzazione |
| 9. Policy Gradient | policy softmax lineare e REINFORCE |

Il self-play in giochi a somma zero e il pool di avversari sono estensioni
rispetto al materiale principale della dispensa. Devono essere introdotti con
chiarezza, senza attribuire loro garanzie teoriche non dimostrate.

Non è necessario usare tutti gli algoritmi del corso. Il progetto approfondisce
pochi concetti coerenti:

- informazione parziale;
- memoria;
- ritorni Monte Carlo;
- policy parametrica;
- policy gradient;
- self-play non stazionario.

## 19. Cosa il progetto non afferma

Il progetto non afferma:

- di aver calcolato la policy ottimale della Briscola;
- di aver raggiunto un equilibrio di Nash;
- di garantire la vittoria contro qualsiasi avversario;
- di aver costruito un belief state esatto;
- di aver modellato lo stile psicologico dell'avversario;
- che battere la snapshot precedente dimostri un miglioramento generale;
- che poche vittorie contro esseri umani costituiscano evidenza statistica.

L'affermazione finale dovrà avere una forma simile a:

> Abbiamo addestrato tramite population-based self-play una policy per la
> Briscola a due che usa la propria mano e la storia pubblica della partita.
> Ne abbiamo valutato robustezza, memoria e generalizzazione contro policy
> congelate e avversari non utilizzati negli aggiornamenti.

## 20. Rischi metodologici

| Rischio | Conseguenza | Mitigazione |
|---|---|---|
| Aggiornare entrambi i ruoli con gli stessi parametri | cancellazione o instabilità del gradiente | learner aggiornato, opponent congelato |
| Opponent solo latest | cicli e overfitting | pool di snapshot |
| Stato con carte nascoste | information leakage | API separate per stato interno e osservazione |
| Feature insufficienti | policy incapace di usare la storia | ablation e feature di interazione |
| Reward terminale sparsa | alta varianza | reward equivalente per presa o baseline |
| Pochi seed | conclusioni dominate dalla fortuna | paired deals e intervalli di confidenza |
| Valutazione solo contro sé stesso | progresso non interpretabile | random, greedy, euristica e pool test |
| Reward manuale sulle briscole | strategia imposta dagli autori | briscola come feature, non penalità |
| Pool crescente senza controllo | costo e ridondanza | limite e regola di conservazione fissati |
| Training instabile | checkpoint apparentemente migliori per caso | evaluation suite separata |

## 21. Fasi di sviluppo

### Fase 0 - Specifica delle regole

Produrre una specifica unica delle regole implementate:

- rappresentazione delle 40 carte;
- ordine e valore;
- determinazione della presa;
- ordine di pesca;
- trattamento della briscola scoperta;
- condizione terminale;
- gestione del pareggio a 60 punti.

Non si procede al training finché le regole non sono congelate.

### Fase 1 - Motore del gioco

Implementare:

- inizializzazione da seed;
- distribuzione delle carte;
- azioni legali;
- risoluzione della presa;
- pesca;
- aggiornamento dei punti;
- termine dell'episodio;
- replay deterministico.

Test obbligatori:

- nessuna carta duplicata;
- tutte le carte vengono giocate;
- somma finale uguale a 120;
- vincitore della presa corretto per ogni combinazione rilevante;
- action mask corretto;
- riproducibilità del seed.

### Fase 2 - Separazione stato/osservazione

Costruire due interfacce:

- stato interno completo, usato soltanto dall'ambiente;
- osservazione del giocatore, priva di informazioni nascoste.

Scrivere test specifici contro information leakage.

### Fase 3 - Baseline

Implementare:

- random;
- greedy di presa;
- una euristica semplice.

Produrre la prima matrice degli scontri. Questa fase verifica che il motore
generi risultati plausibili.

### Fase 4 - Policy parametrica

Implementare:

- funzione \(\phi(M,a)\);
- action masking;
- softmax lineare;
- campionamento stocastico;
- modalità greedy di valutazione.

Controllare manualmente le feature su stati di esempio.

### Fase 5 - REINFORCE contro opponent congelato

Prima del self-play con pool:

1. congelare una policy random;
2. addestrare il learner contro tale policy;
3. verificare che il ritorno medio migliori;
4. verificare che il learner batta la baseline random su seed nuovi.

Questa fase valida la pipeline di apprendimento.

### Fase 6 - Self-play con snapshot

Implementare:

- checkpoint periodici;
- pool degli avversari;
- campionamento uniforme;
- alternanza dei posti;
- aggiornamento del solo learner;
- logging di reward, gradiente e policy.

### Fase 7 - Esperimenti

Eseguire:

- latest-only contro pool storico;
- memoria ridotta contro memoria completa;
- più seed indipendenti;
- matrice degli scontri;
- valutazione contro baseline mai usate per gli aggiornamenti.

### Fase 8 - Interpretazione

Analizzare:

- quando la policy usa le briscole;
- quali feature hanno peso elevato;
- come cambia il comportamento nella fase finale;
- se esistono cicli tra snapshot;
- se il pool migliora il worst-case;
- casi concreti in cui la memoria cambia l'azione.

### Fase 9 - Interfaccia umana

Soltanto dopo aver concluso la valutazione automatica:

- realizzare un'interfaccia minima;
- permettere partite umano-agente;
- salvare il replay;
- mostrare probabilità e feature dopo la partita.

## 22. Divisione del lavoro per quattro persone

| Persona | Responsabilità primaria |
|---|---|
| 1 | formalizzazione, regole, motore e test |
| 2 | osservazioni, memoria, feature e policy |
| 3 | REINFORCE, self-play, pool e checkpoint |
| 4 | baseline, esperimenti, statistica e interfaccia |

La divisione non crea quattro sottoprogetti indipendenti. Tutti devono saper
spiegare:

- perché il gioco è parzialmente osservabile;
- perché l'avversario deve essere congelato durante l'aggiornamento;
- cosa ottimizza la reward;
- cosa garantisce e cosa non garantisce il pool;
- come viene valutata la policy.

## 23. Criteri di successo

### Successo minimo

- motore corretto e riproducibile;
- nessuna informazione nascosta fornita alla policy;
- REINFORCE migliora contro una policy congelata;
- policy finale superiore a random;
- risultati con intervalli di confidenza;
- policy e limiti interpretabili.

### Successo completo

- pool storico migliore di latest-only sulla popolazione di test;
- memoria completa migliore della memoria ridotta;
- policy competitiva contro greedy ed euristica;
- matrice degli scontri priva di regressioni evidenti;
- partite umane plausibili e registrabili.

### Risultato negativo valido

Il progetto rimane scientificamente valido se:

- il pool non migliora rispetto a latest-only;
- la policy lineare non supera una buona euristica;
- REINFORCE mostra varianza eccessiva;
- la memoria completa offre un vantaggio limitato.

In tal caso occorre spiegare il risultato attraverso:

- limiti della rappresentazione;
- capacità della policy;
- non stazionarietà del self-play;
- varianza dei ritorni;
- complessità dell'informazione nascosta.

## 24. Perimetro finale

Il nucleo obbligatorio è:

1. Briscola completa uno contro uno;
2. informazione privata correttamente nascosta;
3. memoria della storia pubblica;
4. policy softmax lineare;
5. REINFORCE;
6. learner contro opponent congelato;
7. self-play dalla policy iniziale;
8. pool di snapshot storiche;
9. baseline di valutazione;
10. confronto latest-only/pool e no-memory/memory.

Restano fuori dal nucleo:

- reti neurali profonde;
- PPO, SAC, DQN e actor-critic;
- belief state esatto;
- Counterfactual Regret Minimization;
- ricerca di un equilibrio di Nash;
- opponent modelling;
- reward manuali per conservare le briscole;
- adattamento online allo stile di una persona;
- Briscola a squadre.

Questo perimetro mantiene il progetto ambizioso ma leggibile: il centro non è
la complessità del codice, bensì la traduzione rigorosa del gioco in un
problema di apprendimento con informazione parziale e self-play.
