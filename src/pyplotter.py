import json
import matplotlib.pyplot as plt
import math
import pprint

class AnnoteFinder(object):
    """callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.

    Register this function like this:

    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)
    
    This doesn't really work well on log scale plots
    """

    def __init__(self, xdata, ydata, annotes, ax=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/float(len(xdata)))*4
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/float(len(ydata)))*4
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def __call__(self, event):

        if event.inaxes:

            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                # print(event.xdata, event.ydata)
                for x, y, a in self.data:
                    # print(x, y, a)
                    if ((clickX-self.xtol < x < clickX+self.xtol) and
                            (clickY-self.ytol < y < clickY+self.ytol)):
                        annotes.append(
                            (self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, ax, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            # Hide all other annotations
            for (x0, y0) in self.drawnAnnotations:
                markers = self.drawnAnnotations[(x0, y0)]
                for m in markers:
                    m.set_visible(False)
                self.ax.figure.canvas.draw_idle()
            t = ax.text(x, y, " - %s" % (annote), bbox=dict(facecolor='gray', alpha=0.5))
            m = ax.scatter([x], [y], marker='d', c='r', zorder=10000)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)

#f = open('./data/synthetic_uniform_sphere_n50000_dim16_m400_k1/data/all.json')
#f = open('./data/synthetic_uniform_sphere_n30000_dim16_m300_k10/data/all.json')
#f = open('./data/sift1m_full/data/all.json')
f = open('./data/sift1m_full_k10/data/all.json')
#f = open('./data/synthetic_uniform_sphere_n90000_dim16_m600_k10/data/all.json')
#f = open('./data/synthetic_uniform_sphere_n56000_dim128_m400_k10/data/all.json')
#f = open('./data/synthetic_uniform_sphere_n56000_dim64_m400_k10/data/latest.json')
#f = open('./data/synthetic_uniform_sphere_n56000_dim128_m400_k10/data/some-combined-stuff.json')
#f = open('./data/synthetic_uniform_sphere_n56000_dim128_m400_k10/data/hnsw2_vs_ehnsw2_vs_filterehnsw.json')
#f = open('./data/synthetic_uniform_sphere_n66000_dim64_m400_k10/data/latest.json')

datavec = json.load(f)

# TODO come up with a better name for this
enable_sprinkles = False
sprinkle_param = 'pruning_factor'

engines = set()
for benchdata in datavec:
    #engines.add(benchdata['engine_name'] + "=" + benchdata['param_list']['max_depth'])
    #if benchdata['engine_name'] != "EHNSW Engine 4" and benchdata['engine_name'] != "EHNSW Engine 5":
    #if benchdata['engine_name'] != "EHNSW Engine 5(no e)" and benchdata['engine_name'] != "EHNSW Engine 5" and benchdata['engine_name'] != "EHNSW Engine 4" and benchdata['engine_name'] != "EHNSW Engine 2":
        continue
    if enable_sprinkles:
        sprinkle = ""
        if sprinkle_param in benchdata['param_list']:
            sprinkle = benchdata['param_list'][sprinkle_param]
        engines.add(benchdata['engine_name'] + "=" + sprinkle)
    else:
        engines.add(benchdata['engine_name'])

from colour import Color
cols = list(Color('blue').range_to(Color('red'), len(engines)))

#annotations = {}

fig, ax = plt.subplots()
xall = []
yall = []
annotationsall = []
#for eng in sorted(engines):
for eng, col in zip(sorted(engines), cols):
    x = []
    y = []
    s = []
    #annotations[eng] = []
    for benchdata in datavec:
        #if ('run_improves' not in benchdata['param_list']) or (benchdata['param_list']['run_improves'] != '0'):
        #    continue
        #bd_name = benchdata['engine_name'] + "=" + benchdata['param_list']['max_depth']
        #if eng == benchdata['engine_name']:
        bd_name = benchdata['engine_name']
        if enable_sprinkles:
            sprinkle = ""
            if sprinkle_param in benchdata['param_list']:
                sprinkle = benchdata['param_list'][sprinkle_param]
            bd_name = benchdata['engine_name'] + "=" + sprinkle
        if eng == bd_name:
            xi = benchdata['recall']
            yi = benchdata['time_per_query_ns']
            x.append(xi)
            xall.append(xi)
            #y.append(yi)
            #yall.append(yi)
            y.append(1e9/yi)
            yall.append(1e9/yi)
            s.append(4)
            #annotations[(xi, yi)] = str(benchdata)
            annotationsall.append(pprint.pformat(benchdata))
            # annotations[eng].append(str(benchdata))
    if enable_sprinkles:
        plt.scatter(x, y, s, label=eng, picker=True, color='{}'.format(col))
    else:
        plt.scatter(x, y, s, label=eng, picker=True)

plt.xlabel("Recall")
#plt.ylabel("Query time (ns)")
plt.ylabel("QPS")

#plt.title("Recall-Querytime for k-NN.")
plt.title("Recall-QPS for k-NN.")

plt.yscale("log")
plt.legend()

af = AnnoteFinder(xall, yall, annotationsall, ax=ax)

fig.canvas.mpl_connect('button_press_event', af)
plt.show()
