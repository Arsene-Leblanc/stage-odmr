### stage-odmr
Le but de ce projet est d'utiliser des cristaux organiques pour prendre des mesures précises de champs magnétiques, de micro-ondes, etc. Les cristaux utiliser peuvent décroitres dans un état triplet permettant un contrôle cohérent des populations (T_x, T_y, T_z). Les temps de décroissance varient et permettent de moduler la photoluminescence (PL). 

## Installation 
```bash
git clone https://github.com/Arsene-Leblanc/stage-odmr.git
cd stage-odmr
```

### Stade 1 : croissance des cristaux
Nous utilisons des cristaux de P-Terphényle (PTP) dopés avec du pentacène (PC) comme base pour nos expériences. Le substrat est du verre, nous utilisons des lamelles de microscope ainsi que des substrats carrés en verre. Le PTP est mis en solution dans de l'acétate d'éthyle et agité pendant une vingtaine de minute. Les deux concentration fonctionnant le mieux pour l'instant sont 1mg/mL et 2mg/mL. Le PC est quant à lui dissous *difficilement* dans du dichlorobenzène. Il semple que la présence de groupements chlorés est bénéfique pour sa dissolution. On chauffe ensuite jusqu'à environ 45 degrées C, avec une agitationt très forte (1600 rpm).On arrive à dissoudre une partie du 1mg dans 20mL. Une partie de cette solution est diluée dans la solution de PTP. 
Il suffit alors de mettre nos lames et lamelles dans des béchers et de laisser la solution s'évaporer. Après 24-72h on se retouvre avec des cristaux à la base des substrats. 

Nos premiers cristaux ont été difficiles à faire croitre, toutefois, nous arrivons dorénavant à avoir d'assez bons résultats répétivement. Voici à quoi ressemble nos cristaux en général (à changer) :
![Cristaux de PTP dopés au PC](photos/photo_echantillons/pterph1.jpg)
### Stade 2 : Mesure de la photoluminescence et du spectre en régime continu
Pour tester nos échantillons, nous avons mis en place un simple montage optique. Il est composé d'un LASER vert dde 505nm et de puissance <5mW. Il vise l'échantillon qui est soutenue par une petite pince. La lumière émise par PL est convergée par une lentille de f=25mm puis un filtre (550nm) empêche notre LASER de polluer la mesure. Au bout, une fibre optique de 10um² capte la PL et se rend dans un spectromètre qui analyse notre spectre d'émission.![Montage optique initial](photos/photo_aléatoire/montageoptique.jpg)

### Résultats du spectre de la photoluminescence
Comme nous sommes encore au tout début de notre projet, il est à noter que nos cristaux sont très inégaux et que certains échantillons performent bien mieux que d'autres. Toutefois, nos résultats s'améliorent de manière relativement constante. Voici quelques spectres prit en ordre chronologique du projet. Les temps d'intégration sont les même pour tous les échantillons(150$\mu$ s) sauf le plus récent qui est de 80$\mu$ s puisqu'il saturait le détecteur sinon. 
![Spectre du pentacène, début](photos/photo_spectre/Spectre_du_pentacène_150us_1.png)
