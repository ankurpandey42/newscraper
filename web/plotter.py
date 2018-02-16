"""


"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import numpy as np
np.set_printoptions(precision=3)


def plot(url, name_clean):
    j_name = './static/{}.json'.format(name_clean)

    json_results = json.load(open(j_name))
    results_ = {k: v for k, v in json_results.items()}
    print(results_)

    def get_spectrum(spec, name, colors):
        spec = dict(zip(spec, range(len(spec))))
        y, x = list(
            zip(*sorted(filter(lambda kv: kv[0] in spec, results_.items()), key=lambda kv: spec[kv[0]])))
        ''' remove denoiseing until new baseline is calculated '''
        make_fig(x, y, name, colors)

    sns.set(style='whitegrid', font='Tahoma', font_scale=1.7)

    def label_cleaner(y):
        key = {
            'fakenews': 'fake news',
            'pro-science': 'scientific',
            'extremeright': 'extreme right',
            'extremeleft': 'extreme left',
            'right-center': 'right of center',
            'left-center': 'left of center',
            'very high': 'very high accuracy',
            'high': 'high accuracy',
            'mixed': 'muddled accuracy',
            'low': 'pants on fire!',
        }
        for label in y:
            for k, v in key.items():
                if label == k:
                    label = v.title()

            yield label.title()

    default_cp = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    policic_colors = ["#9c3229", "#C8493A", "#D6837F", "#DCDDDD", "#98B5C6", "#6398C9", "#3F76BB"]
    veracity_colors = ["#444784", "#2F7589", "#29A181", "#7CCB58"]
    charachter_colors = ["#444784", "#7CCB58", "#3976C5", "#02B97C", "#C8493A"]

    print()

    max_val = max([v for k, v in results_.items() if k != 'n_words'])
    print(max_val)
    print()

    def make_fig(x, y, cat, colors='coolwarm_r'):
        color_p = default_cp
        if cat == "Political":
            color_p = policic_colors
        elif cat == "Accuracy":
            color_p = veracity_colors
        elif cat == "Character":
            color_p = charachter_colors

        y = list(label_cleaner(y))

        plt.figure(figsize=(8, 8))
        y_pos = np.arange(len(y))
        # x = np.square(np.asarray(x))
        x = np.asarray(x)
        g = sns.barplot(y=y_pos, x=x, palette=(sns.color_palette(color_p)), orient='h', saturation=.9)
        plt.yticks(y_pos, y)
        plt.title('{} - {}'.format(url, cat))
        plt.xlabel('Neural network estimation')

        plt.xlim(0, max_val)

        plt.savefig(
            './static/{}.png'.format(name_clean + '_' + cat), format='png', bbox_inches='tight', dpi=100)

        plt.clf()

    get_spectrum(
        ['extreme right', 'right', 'right-center', 'center', 'left-center', 'left',
         'extreme left'], 'Political', 'policic_colors')

    get_spectrum(['very high', 'high', 'mixed', 'low', 'unreliable'], 'Accuracy', 'veracity_colors')
    plt.close('all')

    get_spectrum(['conspiracy', 'fake news', 'propaganda', 'pro-science', 'satire', 'hate'], 'Character',
                 'charachter_colors')


if __name__ == '__main__':

    plot(' _test_', 'breitbartcom')
    plot(' _test_', 'msnbccom')
