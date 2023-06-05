import csv
import plotly.graph_objects as go

file = '/home/machine_visoin/rosws/src/runscript/src/newcsv.csv'

time = []
xvalues = []
yvalues = []
zvalues = []

qxvalues = []
qyvalues = []
qzvalues = []
qwvalues = []

with open(file , 'r')  as file :
    reader = csv.reader(file , delimiter='\t')
    next(reader)
    for row in reader :
        time.append(float(row[0]))
        xvalues.append(float(row[1]))
        yvalues.append(float(row[2]))
        zvalues.append(float(row[3]))
        qxvalues.append(float(row[4]))
        qyvalues.append(float(row[5]))
        qzvalues.append(float(row[6]))
        qwvalues.append(float(row[7]))


position_trace = go.Scatter3d(

    x = xvalues , 
    y = yvalues , 
    z = zvalues  ,
    mode = 'markers' ,
    marker=dict(
        size = 5 ,
        color = time ,
        colorscale = 'blues' ,
        symbol = 'circle' ,
        line = dict(width=1) , 
        opacity = 0.8 ,
        colorbar = dict(title='Time')
    ) , 
    hoverinfo = 'text' , 
    text = ['Time: {:.2f}'.format(tim) for tim in time] ,
     name = 'Position' 
)


orientation_trace = go.Scatter3d(
    x = xvalues , 
    y = yvalues , 
    z = zvalues ,
    mode = 'markers' ,
    marker = dict(
        size = 5 ,
        color = 'red' , 
        symbol = 'cross' ,
        line = dict(width=1) ,
        opacity = 0.8
    ) ,
    hoverinfo = 'text' , 
    text = ['Time: {:.2f}'.format(tim) for tim in time] , 
    name = 'Orientation'
     
)

layout = go.Layout(
    title = 'Position and Orientation' ,
    scene = dict(
        xaxis= dict(title='X') , 
        yaxis = dict(title='Y') , 
        zaxis = dict(title='Z')
    )
)


fig = go.Figure(data=[position_trace , orientation_trace] , layout=layout)
fig.show()




