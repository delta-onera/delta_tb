import os
import numpy as np
import torch
import visdom

class VisuVisdom:
    def __init__(self, env_name, **kwargs):
        self.env_name = env_name
        self.win_refs = {}

        self.vis = visdom.Visdom(**kwargs)

    def new_plot(self, title, epoch, value, legend=None):
        if legend is not None:
            self.win_refs[title] = self.vis.line(X=np.array([epoch]),
                                 Y=np.array([value]),
                                 env=self.env_name, opts=dict(
                                                        title=title,
                                                        xlabel='Epochs',
                                                        ylabel=title,
                                                        legend=legend,
                                                        showlegend=True))
        else:
            self.win_refs[title] = self.vis.line(X=np.array([epoch]),
                                 Y=np.array([value]),
                                 env=self.env_name, opts=dict(
                                                        title=title,
                                                        xlabel='Epochs',
                                                        ylabel=title))

    def plot(self, title, epoch, value, legend=None):
        if title in self.win_refs:
            if legend is not None:
                self.vis.line(X=np.array([epoch]),
                             Y=np.array([value]),
                             win=self.win_refs[title], env=self.env_name, update='append',
                             opts=dict(
                                    title=title,
                                    xlabel='Epochs',
                                    ylabel=title,
                                    legend=legend,
                                    showlegend=True))
            else:
                self.vis.line(X=np.array([epoch]),
                             Y=np.array([value]),
                             win=self.win_refs[title], env=self.env_name, update='append',
                             opts=dict(
                                    title=title,
                                    xlabel='Epochs',
                                    ylabel=title))
        else:
            self.new_plot(title, epoch, value, legend)

    def imshow(self, tensor_, title, nb_img_max=8, largeur_max=512, unnormalize=False):
        """
        affiche une ou plusieurs images contenues dans un torch.tensor ou une liste de torch.tensor
        """
        if type(tensor_) == list:
            tensor_ = torch.stack(tensor_)

        tensor = tensor_.clone()
        if unnormalize:
            for i in range(tensor.size(0)):
                tensor[i] -= tensor[i].min()
                tensor[i] *= 255 / tensor[i].max()

        if tensor.ndimension() == 4:
            n, c, h, w = tensor.size()
            if n > nb_img_max:
                n = nb_img_max
                tensor = tensor[:n]
            nrow = int(np.ceil(np.sqrt(n)))
            width = largeur_max
            height = h / w * largeur_max
            opts = dict(title=title, width=width, height=height)
            if not title in self.win_refs:
                self.win_refs[title] = self.vis.images(tensor, nrow=nrow,
                                                        env=self.env_name, opts=opts)
            else:
                self.vis.images(tensor, win=self.win_refs[title],
                                nrow=nrow, env=self.env_name, opts=opts)
        elif tensor.ndimension() == 3:
            c, h, w = tensor.size()
            width = largeur_max
            height = h / w * largeur_max
            opts = dict(title=title, width=width, height=height)
            if not title in self.win_refs:
                self.win_refs[title] = self.vis.image(tensor, env=self.env_name,
                                                        opts=opts)
            else:
                self.vis.image(tensor, win=self.win_refs[title],
                                    env=self.env_name, opts=opts)
        else:
            raise TypeError("tensor must be a 3 or 4-dim tensor or a list of 3-dim tensors")

    def heatmap(self, tensor, title, **opts):
        """
        affiche une seule heatmap contenue dans un np.array 2D ou torch tensor 2D

        opts.colormap : colormap (string; default = 'Viridis')
        opts.xmin : clip minimum value (number; default = X:min())
        opts.xmax : clip maximum value (number; default = X:max())
        opts.columnnames: table containing x-axis labels
        opts.rownames : table containing y-axis labels
        opts.layoutopts : dict of any additional options that the graph backend
                        accepts for a layout. For example layoutopts =
                                        {'plotly': {'legend': {'x':0, 'y':0}}}.

        """
        h, w = tensor.shape
        opts['title'] = title
        if not title in self.win_refs:
            self.win_refs[title] = self.vis.heatmap(tensor[::-1,:], env=self.env_name,
                                                    opts=opts)
        else:
            self.vis.heatmap(tensor[::-1,:], win=self.win_refs[title],
                                env=self.env_name, opts=opts)


    def save(self):
        """
        Sauvegarde l'environnement dans ~/.visdom
        """
        self.vis.save([self.env_name])
