# Regn input size
En af de sværeste del er at beregne hvad skal input størrelsen af den første fully connected lag er. Især når der kommer mange convolution og max pooling lag:
- En ide vil være at bruge code judge og bedde deltagerne til at skrive funktionen ned der beregner hvad størrelsen bliver. Så sender vi en del højder og bredde og forventer vi et svar.
- En anden ide er at give dem et pytorch netværk klasse, hvor input størrelsen af den første fc lag mangler, og de skal sætte det ind og se om modellen kører. Problemet her er at error beskeden viser hvad størrelsen skal være.
- En bedre ide kan være at vi giver dem en pytorch klasse istedet for en sekvens af lag i codejudge og spørger dem at skrive den rigtigt funktion ind.

## Regler:
Conv kernel = x -> Størrelsen - (x-1)
Conv stride = x -> Størrelsen // x
Conv padding = x -> Størrelsen + 2x
Order of operations: Padding then Kernel then Stride
MaxPool kernel = x -> Størrelsen // x

Opgave liste:
## Opgave 1
```py
conv(input_filtre=1, output_filtre=128, kerne=3, stride=1, padding=0)
flatten()
fc1(input_størrelse=??)
```

## Opgave 2
```py
Conv(input_filtre=1, output_filtre=32, kerne=5, stride=1, padding=0)
Conv(input_filtre=32, output_filtre=64, kerne=3, stride=1, padding=0)
flatten()
fc1(input_størrelse=??)
```

## Opgave 3
```py
conv(input_filtre=1, output_filtre=32, kerne=23, stride=1, padding=0)
flatten()
fc1(input_størrelse=??)
```

## Opgave 4
```py
conv(input_filtre=1, output_filtre=32, kerne=3, stride=1, padding=0)
maxpool(kerne=2)
flatten()
fc1(input_størrelse=??)
```

## Opgave 5
```py
conv(input_filtre=1, output_filtre=32, kerne=3, stride=1, padding=1)
conv(input_filtre=32, output_filtre=64, kerne=3, stride=1, padding=1)
flatten()
fc1(input_størrelse=??)
```

## Opgave 6
```py
conv(input_filtre=1, output_filtre=32, kerne=3, stride=1, padding=0)
maxpool(kerne=2)
conv(input_filtre=32, output_filtre=64, kerne=3, stride=1, padding=0)
flatten()
fc1(input_størrelse=??)
```

## Opgave 7
```py
conv(input_filtre=1, output_filtre=32, kerne=3, stride=1, padding=1)
maxpool(kerne=2)
conv(input_filtre=32, output_filtre=64, kerne=3, stride=1, padding=1)
flatten()
fc1(input_størrelse=??)
```

## Opgave 8
```py
conv(input_filtre=1, output_filtre=32, kerne=5, stride=1, padding=2)
maxpool(kerne=2)
conv(input_filtre=32, output_filtre=64, kerne=3, stride=1, paddins=0)
conv(input_filtre=64, output_filtre=128, kerne=3, stride=1, padding=0)
maxpool(kerne=2)
flatten()
fc1(input_størrelse=??)
```

## Opgave 9
```py
conv(input_filtre=1, output_filtre=32, kerne=3, stride=2, padding=1)
conv(input_filtre=32, output_filtre=64, kerne=3, stride=2, padding=1)
flatten()
fc1(input_størrelse=??)
```

# Ekstra svær opgaver
Implementere selv NN funktioner. Fully connected, convolutional og max pooling. Implementerer også RELU, RELU6 og leaky RELU. Jo mere opgaver jo bedre :3

