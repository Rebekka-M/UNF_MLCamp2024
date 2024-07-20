# Velkommen til UNFs tegn og gæt konkurrence
Kender du til spillet [Quick, Draw!](https://quickdraw.withgoogle.com/)? I dette modul skal du træne din/jeres egen algoritme til at genkende tegninger ("doodles") lige som den algoritme du kan se i spillet. De algoritmer vi kommer til at bygge er neurale netværk, ligesom dem i lige har lært om.

## Introduktion
Jeres opgave er:
- At definere model architecturen
- Finde de rigtige hyperparametre
- Træne modellen med forskellige kombinationer af model architecturer og hyperparametre

Når I har fundet en model i er glad for, kan i dele den med det faglige team (nærmere info følger om hvordan) og til sidst afholder vi en tegn og gæt konkurrence med alle modellerne for at se hvem der har bygget den bedste tegn og gæt model.

Herunder følger en dokumentation på koden basen du skal bruge under konkurrencen

## Dokumentation
Kodebasen består af følgende filer:
```python
QuickDraw     
├── data                    # Folder til at gemme en lokal version af de data filer vi bruger til træning
│   └── _data_static.py     # Giver labels på de klasser vi træner på (Niks pille ;) )
├── saved_models            # Folder hvor jeres gemte modeller i har trænet ligger i
├── app.py                  # Application i kan bruge til at teste gemte modeller lokalt
├── get_data.py             # Definerer jeres data loader til træning
├── model.py                # ~ Her skal I definere model arkitekturen og vælge hyperparametre ~
├── options.py              # ~ Her kan O tilføje nye hyperparametre, hvis I vil bruge noget mere eksotisk ~
├── train.py                # Træningsproceduren (Niks pille ;) )
└── README.md               # Denne fil
```
Det er generalt set en god ide at læse kodebasen igennem før i går igang med at bygge og træne modeller. Hvis i har spørgsmål kan i altid tage fat i faglig for at få hjælp og forklaringer.

## Arbejdsgang
I kommer mest af alt til at arbejde fra 2 filer: `model.py` og `options.py`. Her følger lidt mere info om hvordan i kan bruge kodebasen til træning og evaluering af modeller.

### 1. Naviger til QuickDraw mappen
Åben jeres terminal og skriv
```bash
cd 2.NN/QuickDraw/
```

### 2. Træn model
Efter i har defineret en model i `model.py` filen, kan i træne den ved at følgende kommando i terminalen:
```bash
python model.py
```

Den kommando kører alt i `model.py`filen der ligger efter `if __name__ == "__main__":` linjen i filen, der gør følgende:
1. Henter dataen
2. Definerer hyperparametre
3. Definerer modellen
4. Træner modellen (og logger den med mlflow)
5. Gemmer modellen i `saved_models`

### 3. Sammenlign modeller
Eftersom vi logger modellerne i `mlflow` under træning, kan vi når vi har bygget flere modeller sammenligne med visuel:
```bash
mlflow server
```
Hvis i derfra navigerer til `http://127.0.0.1:5000` (jeres localhost) får i et dashboard i kan bruge til at sammenligne forskellige modeller. Denne er tilgængig så længe denne kommando er aktiv

### 4. Test jeres modeller
Hvis I har en model i gerne vil teste ydeligere, kan i teste den i med live tegninger i `app.py`:
```bash
streamlit run app.py
```

# :sparkles: Vi ønsker held og lykke med konkurrencen :sparkles:




