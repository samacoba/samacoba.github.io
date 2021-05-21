---
layout: post
title: 新型コロナが現れた世界に適応する
category: blog
tags: はてな COVID19
---

### 新型コロナの弱点

前回の[シミュレーション](https://samacoba.github.io/20200229covid_cluster/)と同様に

* 5人の感染者の4人は他の人にあまり感染させてない
* 5人の内の1人が他の多くの人に感染させているようである

のようなスーパースプレッダが発生するケースを想定して、それに対する弱点について考えてみたいと思います。

専門家会議では風通しの悪い密閉空間など環境要因を強調されていますが、スプレッダを仮定するとコミュニティ**数**の制限による効果も大きいと考えられます。

ここで、例えば3つのコミュニティ（職場、家庭、趣味）を考えて、Aさんはこの3つのコミュニティに属しており、3つのコミュニティにはそれぞれA以外に5人いるとします。
先に職場のFさんが他のコミュニティで感染し、その後F⇒Aさんに感染したという場合を考えます。

ケース➀のスーパースプレッダなしでは、たまたまF⇒Aさんだけに感染したと仮定した場合、その後Aさん⇒職場、家庭、趣味の合計14人に感染させる可能性があります。
ケース➁のスーパースプレッダありでは、Fはスーパースプレッダであるため、F⇒E,D,C,B,Aさん全員に感染して、その後Aさんのみがスプレッダになったと仮定すると、その後Aさんは⇒家庭、趣味の合計10人に感染させる可能性があります。
（職場のE,D,C,Bさんはすでに感染しているので、もう一度感染することはない）

![imgae](/images/20200305-01.PNG)

ここで、Aさんが趣味のコミュニティ（ジム、ライブなど）に行かないようにし、コミュニティ数を3 ⇒ 2に制限した場合を考えます。

* ケース➀ではAさんからの感染可能性のある人数が14人 ⇒ 9人に減る(36%減)
* ケース➁ではAさんからの感染可能性のある人数が10人 ⇒ 5人に減る(50%減)

従って、スーパースプレッダが発生する条件では、**コミュニティ「数」を制限する効果がより大きくなる**と考えられます。

### 新型コロナに適応する

個人差はあると思いますが、普段の生活の中で、おそらく接触を減らすのが難しい順に、家庭 ＞ 職場 ＞ 趣味 となり、家庭と職場など最低２つのコミュニティを維持できれば最低限の生活は成り立つ場合も多いかと思います。

仮にコミュニティ「数」を２つ以内に制限するなどで、もし基本再生産数R0を１未満に抑えることができれば、日本における新型コロナを収束させることは可能だと思います。

とはいえ、現在のヨーロッパや米国の発生状況を見る限り、今後世界的に新型コロナが長期間流行する可能性が高く、鎖国して日本の新型コロナの発生を**ゼロにし、元の生活に戻る**ということも難しいと考えられます。

では、「趣味を生きがいにしていた人はどうするのか」や、「ライブのイベントや観光などで生計を立ててる人はどうするのか」など思う方がいるかもしれません。

このあたりは災害や急激な環境変化と同じで、**起きてしまった現状**に対して、今までの生活習慣をそのまま続けることはあきらめて、新型コロナが現れた世界に**適応する**しかないと思います。

幸いネットワークが発達した現代、オンラインでのコミュニティを楽しむことは十分可能で、趣味もオンラインにシフトさせていくことで適応するのも一つの方法だと思います。今後は一人ひとりがどうやってこの環境に適応するかを考えていくことが重要になってくると思います。







