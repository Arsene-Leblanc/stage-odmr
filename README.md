# stage-odmr
Le but de ce projet est d'utiliser des cristaux organiques pour prendre des mesures précises de champs magnétiques, de micro-ondes, etc. Les cristaux utiliser peuvent décroitres dans un état triplet permettant un contrôle cohérent des populations (T_x, T_y, T_z). Les temps de décroissance varient et permettent de moduler la photoluminescence (PL). 

## Installation 
```bash
git clone https://github.com/Arsene-Leblanc/stage-odmr.git
cd stage-odmr
```

### Stade 1 : croissance des cristaux
Nous utilisons des cristaux de P-Terphényle (PTP) dopés avec du pentacène (PC) comme base pour nos expériences. Le substrat est du verre, nous utilisons des lamelles de microscope ainsi que des substrats carrés en verre. Le PTP est mis en solution dans de l'acétate d'éthyle et agité pendant une vingtaine de minute. Les deux concentration fonctionnant le mieux pour l'instant sont 1mg/mL et 2mg/mL. Le PC est quant à lui dissous *difficilement* dans du dichlorobenzène. Il semple que la présence de groupements chlorés est bénéfique pour sa dissolution. On chauffe ensuite jusqu'à environ 45 degrées C, avec une agitationt très forte (1600 rpm).On arrive à dissoudre une partie du 1mg dans 20mL. Une partie de cette solution est diluée dans la solution de PTP. 
Il suffit alors de mettre nos lames et lamelles dans des béchers et de laisser la solution s'évaporer. Après 24-72h on se retouvre avec des cristaux à la base des substrats. 

### Stade 2 : Mesure de la photoluminescence et du spectre en régime continu
Pour tester nos échantillons, nous avons mis en place un simple montage optique. Il est composé d'un LASER vert dde 505nm et de puissance <5mW. Il vise l'échantillon qui est soutenue par une petite pince. La lumière émise par PL est convergée par une lentille de f=25mm puis un filtre (550nm) empêche notre LASER de polluer la mesure. Au bout, une fibre optique de 10um² capte la PL et se rend dans un spectromètre qui analyse notre spectre d'émission.![Montage optique initial](photos/photo_aléatoire/montageoptique.jpg)
