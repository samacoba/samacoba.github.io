---
layout: post
title: 新型コロナから自分の病院を守る相談システムを考えてみる
category: blog
tags: COVID19
---


新型コロナが拡大の中、厚生省は熱が4日以上続く方は帰国者・接触者相談センターに相談と言いますが、患者が「昨日高熱が出たのでインフルエンザだと思う」と一般病院を受診しても、新型コロナのリスクは十分あるわけで、完全な分離は現実には不可能な状況です。

となると、各一般病院は自分で自分の病院を守るしかありません。

一番確実な方法は外来を全てストップすることですが、全ての病院がそれをすると新型コロナ以外の病気を救えなくなってしまいます。

そこで、新型コロナのリスクなるべく下げて、診察できるようなシステムが必要になってくると思います。

例えば、

➀電話相談なしでの初診外来は受け付けない

➁午前を「電話相談」、午後を「外来」などに時間帯を分ける

➂電話相談の結果、

・相談した病院で診察可能な場合、午後に診察に来てもらう

・自宅療養が望ましい場合、そのまま自宅で様子をみてもらう

・新型コロナ感染の高リスクかつ診察が必要な場合、専門病院を紹介し、相談した病院での受診を認めない

患者にとっては直接医者と話して、医者に「自宅で様子をみれば大丈夫」など判断をしてくれるなら安心すると思うので、電話相談は医者が直接行う方が良いのではないかと思います。

また、電話相談は「診断」にあたらないことにすれば、法的な制約は少なく各病院の判断で自由に導入できるのではないかと思います。
（電話相談分の医療費は回収できないかもしれないですが、現場が完全ストップするよりよいかと思います）

ここまではアナログ的に電話窓口を設置し、各病院で導入することも可能ですが、今の時代、もう少し効率のいいオンラインシステムをササっと作った方がいいと思います。



例えば、

* 都道府県くらいの単位で一つのwebサイトをつくる
* 患者や医者はスマートフォンやPCでアクセスする
* 各病院単位で登録し、サイト内に窓口を作成
* 患者は事前に氏名・生年月日・住所・連絡先などをサイトに登録
* 患者は相談したい病院を予約し、問診表を入力
* 一般病院は相談予約リストから順に患者情報を照会しながら、医者側から患者に電話をかけて電話相談する
* 相談の結果、新型コロナ感染の高リスクかつ診察が必要な場合、一般病院から専門病院へ依頼し、患者IDを伝える
* 専門病院は患者情報を照会し、来院方法を患者に伝える
* 患者は専門病院の指示に従い、専門病院を受診する

![imgae](/images/20200222-01.PNG)

![imgae](/images/20200222-02.PNG)


なるべく最低限の機能に絞って、早く立ちあげることを優先して、都度機能を修正・追加する形がいいかと思います。
技術的には難しくないはずで、インフラもクラウド使えば、1週間程度で立ち上げてくれる凄腕のエンジニアや太っ腹の企業らがきっと日本にはいるはずです。
自分自身Webやサーバーのエンジニアでなく、医療関係者でもないので、適当なことしかいえませんが、この新型コロナに対し、医療崩壊を少しでも抑えるための議論のきっかけになれば幸いと思います。




















