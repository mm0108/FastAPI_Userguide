{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPogHy/KmNIgl1NF46trAN3",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mm0108/FastAPI_Userguide/blob/main/sampling_tips.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CtvfcBSFKFXK",
        "outputId": "787c9eb7-c46d-4a32-eb13-e2e098af87cd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "단순 무작위 추출 크기: 30\n"
          ]
        }
      ],
      "source": [
        "import seaborn as sns\n",
        "\n",
        "# 데이터 로드\n",
        "tips = sns.load_dataset('tips')\n",
        "\n",
        "# 30개의 데이터를 무작위로 추출\n",
        "random_sample = tips.sample(n=30, random_state=42)\n",
        "print(f\"단순 무작위 추출 크기: {len(random_sample)}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# 'sex'(성별) 비율을 유지하면서 전체의 20%를 추출\n",
        "stratified_sample, _ = train_test_split(\n",
        "    tips,\n",
        "    test_size=0.2,\n",
        "    stratify=tips['sex'],\n",
        "    random_state=42\n",
        ")\n",
        "\n",
        "print(\"층화 추출 내 성별 비율:\\n\", stratified_sample['sex'].value_counts(normalize=True))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LMqeZp1zKK-c",
        "outputId": "fa213641-26c3-4a4d-d13a-a044b98e65a5"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "층화 추출 내 성별 비율:\n",
            " sex\n",
            "Male      0.641026\n",
            "Female    0.358974\n",
            "Name: proportion, dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "# 표본 간격 설정 (예: 10번째 데이터마다 추출)\n",
        "k = 10\n",
        "start_index = np.random.randint(0, k) # 시작점 무작위 선택\n",
        "\n",
        "# 인덱싱을 이용한 계통 추출\n",
        "systematic_sample = tips.iloc[start_index::k]\n",
        "print(f\"계통 추출 크기: {len(systematic_sample)}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-PdVQ8vbKnMi",
        "outputId": "1fb2690b-7f5a-4bcf-eb2b-12a9b2c8fe1c"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "계통 추출 크기: 24\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "import pandas as pd\n",
        "\n",
        "# dataset yuklash\n",
        "tips = sns.load_dataset(\"tips\")\n",
        "\n",
        "# simple random sampling\n",
        "sample_random = tips.sample(n=50, random_state=1)\n",
        "\n",
        "print(sample_random)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YWB3jmIiKZ7D",
        "outputId": "21842113-99cb-4d62-de3d-e32dbed2ddae"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     total_bill    tip     sex smoker   day    time  size\n",
            "67         3.07   1.00  Female    Yes   Sat  Dinner     1\n",
            "243       18.78   3.00  Female     No  Thur  Dinner     2\n",
            "206       26.59   3.41    Male    Yes   Sat  Dinner     3\n",
            "122       14.26   2.50    Male     No  Thur   Lunch     2\n",
            "89        21.16   3.00    Male     No  Thur   Lunch     2\n",
            "218        7.74   1.44    Male    Yes   Sat  Dinner     2\n",
            "58        11.24   1.76    Male    Yes   Sat  Dinner     2\n",
            "186       20.90   3.50  Female    Yes   Sun  Dinner     3\n",
            "177       14.48   2.00    Male    Yes   Sun  Dinner     2\n",
            "4         24.59   3.61  Female     No   Sun  Dinner     4\n",
            "220       12.16   2.20    Male    Yes   Fri   Lunch     2\n",
            "226       10.09   2.00  Female    Yes   Fri   Lunch     2\n",
            "116       29.93   5.07    Male     No   Sun  Dinner     4\n",
            "107       25.21   4.29    Male    Yes   Sat  Dinner     2\n",
            "170       50.81  10.00    Male    Yes   Sat  Dinner     3\n",
            "241       22.67   2.00    Male    Yes   Sat  Dinner     2\n",
            "181       23.33   5.65    Male    Yes   Sun  Dinner     2\n",
            "51        10.29   2.60  Female     No   Sun  Dinner     2\n",
            "27        12.69   2.00    Male     No   Sat  Dinner     2\n",
            "240       27.18   2.00  Female    Yes   Sat  Dinner     2\n",
            "219       30.14   3.09  Female    Yes   Sat  Dinner     4\n",
            "34        17.78   3.27    Male     No   Sat  Dinner     2\n",
            "93        16.32   4.30  Female    Yes   Fri  Dinner     2\n",
            "183       23.17   6.50    Male    Yes   Sun  Dinner     4\n",
            "118       12.43   1.80  Female     No  Thur   Lunch     2\n",
            "117       10.65   1.50  Female     No  Thur   Lunch     2\n",
            "106       20.49   4.06    Male    Yes   Sat  Dinner     2\n",
            "73        25.28   5.00  Female    Yes   Sat  Dinner     2\n",
            "38        18.69   2.31    Male     No   Sat  Dinner     3\n",
            "210       30.06   2.00    Male    Yes   Sat  Dinner     3\n",
            "202       13.00   2.00  Female    Yes  Thur   Lunch     2\n",
            "44        30.40   5.60    Male     No   Sun  Dinner     4\n",
            "62        11.02   1.98    Male    Yes   Sat  Dinner     2\n",
            "238       35.83   4.67  Female     No   Sat  Dinner     3\n",
            "85        34.83   5.17  Female     No  Thur   Lunch     4\n",
            "224       13.42   1.58    Male    Yes   Fri   Lunch     2\n",
            "39        31.27   5.00    Male     No   Sat  Dinner     3\n",
            "31        18.35   2.50    Male     No   Sat  Dinner     4\n",
            "18        16.97   3.50  Female     No   Sun  Dinner     3\n",
            "132       11.17   1.50  Female     No  Thur   Lunch     2\n",
            "119       24.08   2.92  Female     No  Thur   Lunch     4\n",
            "19        20.65   3.35    Male     No   Sat  Dinner     3\n",
            "91        22.49   3.50    Male     No   Fri  Dinner     2\n",
            "69        15.01   2.09    Male    Yes   Sat  Dinner     2\n",
            "33        20.69   2.45  Female     No   Sat  Dinner     4\n",
            "90        28.97   3.00    Male    Yes   Fri  Dinner     2\n",
            "35        24.06   3.60    Male     No   Sat  Dinner     3\n",
            "11        35.26   5.00  Female     No   Sun  Dinner     4\n",
            "29        19.65   3.00  Female     No   Sat  Dinner     2\n",
            "0         16.99   1.01  Female     No   Sun  Dinner     2\n"
          ]
        }
      ]
    }
  ]
}