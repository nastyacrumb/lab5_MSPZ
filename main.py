import tkinter as tk
from tkinter import ttk
import numpy
import scipy.special
from tkinter import messagebox as mb
import time
import pickle

class ExampleApp(tk.Tk):
    def __init__(self):
        self.input_nodes = 64
        self.hidden_nodes = 10
        self.out_nodes = 5
        self.lern_node = 0.5
        self.example = [
                                        "Не похоже ни на что",
                                        "Y",
                                        "V",
                                        "H",
                                        "P"]
        tk.Tk.__init__(self)
        self.t = SimpleTable(self, 8, 8)
        self.t.pack(side="left", fill="x")
        btn = tk.Button(text="Добавить в обучающую выборку", command=self.t.dump_data)
        btn7 = tk.Button(text="Добавить в тестовую выборку", command=self.t.dump_data_test)

        btn3 = tk.Button(text="Обучить сеть", command=self.train_net)
        btn6 = tk.Button(text="Распознать образ", command=self.check_data)
        btn4 = tk.Button(text="Сбросить все результаты обучения", command=self.reinit_net)
        btn5 = tk.Button(text="Сохранить и выйти", command=self.quits)
        btn2 = tk.Button(text="Очистить", command=self.t.clear)
        btn8 = tk.Button(text="Провести тестирование на тестовой выборке", command=self.start_test_net)

        btn2.pack(side="bottom", fill="x")
        self.combo1 = ttk.Combobox(values=[
                                        "Не похоже ни на что",
                                        "Y",
                                        "V",
                                        "H",
                                        "P"])
        self.combo1.current(0)

        btn7.pack(side="bottom",fill="x")
        btn.pack(side="bottom", fill="x")
        self.combo1.pack(side="bottom", fill="x")
        btn3.pack(side="top", fill="x")
        btn8.pack(side="top", fill="x")
        btn4.pack(side="top", fill="x")
        btn6.pack(side="top", fill="x")
        btn5.pack(side="top", fill="x")
        self.reinit_net()
       # self.start_test_net()

    def init_net(self):
        self.input_nodes = 64
        self.out_nodes = 5
        self.hidden_nodes = 10
        self.lern_node = 0.5

    def creat_net(self, input_nodes, hidden_nodes, out_nodes, ):
        self.input_hidden_w = (numpy.random.rand(hidden_nodes, input_nodes) - 0.5)
        self.hidden_out_w = (numpy.random.rand(out_nodes, hidden_nodes) - 0.5)

    def fun_active(self,x):
        return scipy.special.expit(x)

    def query(self, input_hidden_w, hidden_out_w, inputs_list):
        inputs_sig = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)
        hidden_out = self.fun_active(hidden_inputs)
        final_inputs = numpy.dot(hidden_out_w, hidden_out)
        final_out = self.fun_active(final_inputs)
        return final_out

    def treyn(self, targget_list, input_list, input_hidden_w, hidden_out_w, lern_node):
        targgets = numpy.array(targget_list, ndmin=2).T
        inputs_sig = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(input_hidden_w, inputs_sig)
        hidden_out = self.fun_active(hidden_inputs)
        final_inputs = numpy.dot(hidden_out_w, hidden_out)
        final_out = self.fun_active(final_inputs)
        out_errors = targgets - final_out
        hidden_errors = numpy.dot(hidden_out_w.T, out_errors)
        hidden_out_w += lern_node * numpy.dot((out_errors * final_out * (1 - final_out)), numpy.transpose(hidden_out))
        input_hidden_w += lern_node * numpy.dot((hidden_errors * hidden_out * (1 - hidden_out)),
                                                numpy.transpose(inputs_sig))
        return hidden_out_w, input_hidden_w

    def train_set(self,hidden_out_w, input_hidden_w):

        data_file = open('train.csv', 'r')
        trening_list = data_file.readlines()
        data_file.close()
        hidden_out_w_r = hidden_out_w
        input_hidden_w_r = input_hidden_w
        for record in trening_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) * 0.99) + 0.01
            targets = numpy.zeros(5) + 0.01
            targets[int(all_values[0])] = 0.99
            hidden_out_w_r, input_hidden_w_r = self.treyn(targets, inputs, input_hidden_w_r, hidden_out_w_r, self.lern_node)
        return hidden_out_w, input_hidden_w,len(trening_list)

    def test_spec_val(self):
        data_file = open('tmp.csv', 'r')
        test_list = data_file.readlines()
        data_file.close()
        test = []
        for record in test_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) * 0.99) + 0.01
            out_session = self.query(self.input_hidden_w, self.hidden_out_w, inputs)
            return numpy.argmax(out_session)

    def test_set(self, hidden_out_w, input_hidden_w):
        start_time = time.time()
        data_file = open('test.csv', 'r')
        test_list = data_file.readlines()
        data_file.close()
        test = []
        for record in test_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) * 0.99) + 0.01
            out_session = self.query(input_hidden_w, hidden_out_w, inputs)
            if int(all_values[0]) == numpy.argmax(out_session):
                test.append(1)
            else:
                test.append(0)
        print(len(test))
        test = numpy.asarray(test)
        mb.showinfo('Результат', 'Эфективность сети % ='+str((test.sum() / test.size) * 100)+'\nВсего было '+str(test.size)+" тестов"+"\nВремя выполнения (с):"+str(time.time()-start_time))


    def check_data(self):
        start_time = time.time()
        self.t.get_data_table()
        res = self.test_spec_val()
        mb.showinfo("Результат распозванавания", "Было распознано: "+self.example[res]+"\nВремя выполнения (с):"+str(time.time()-start_time))

    def start_test_net(self):
            self.test_set(self.hidden_out_w, self.input_hidden_w)

    def reinit_net(self):
        self.init_net()
        self.creat_net(self.input_nodes, self.hidden_nodes, self.out_nodes)

    def train_net(self):
        start_time = time.time()
        self.hidden_out_w, self.input_hidden_w,size = self.train_set(self.hidden_out_w, self.input_hidden_w)
        mb.showinfo("Результат обучения", "Обучение выполнено успешно!\nКол-во элементов в обучающей выборке:"+str(size)+"\nОбучение заняло (с):" + str(time.time() - start_time))
    def quits(self):
        self.quit()


class SimpleTable(tk.Frame):
    def __init__(self, parent, rows=10, columns=2):
        self.example = {"Y": "1",
                        "V": "2",
                        "H": "3",
                        "P": "4",
                        "Не похоже ни на что": "0"
        }
        tk.Frame.__init__(self, parent, background="black")
        self._widgets = []
        self.parent = parent
        for row in range(rows):
            current_row = []
            for column in range(columns):
                label = tk.Label(self, name="%s/%s" % (row, column),
                                 borderwidth=0, width=8, height=4, bg="white")
                label.bind('<Button-1>', self.click)
                label.grid(row=row, column=column, sticky="nsew", padx=1, pady=1)
                current_row.append(label)
            self._widgets.append(current_row)

        for column in range(columns):
            self.grid_columnconfigure(column, weight=1)

    def click(self, e):
        list_coord = e.widget._name.split('/')
        colour = self._widgets[int(list_coord[0])][int(list_coord[1])]['bg']
        if colour != "black":
            self._widgets[int(list_coord[0])][int(list_coord[1])]['bg'] = "black"
        else:
            self._widgets[int(list_coord[0])][int(list_coord[1])]['bg'] = "white"

    def set(self, row, column, value):
        widget = self._widgets[row][column]
        widget.configure(text=value)

    def clear(self):
        for i in self._widgets:
            for j in i:
                j['bg'] = "white"

    def get_data_table(self):
        res = []
        for i in self._widgets:
            for j in i:
                if j['bg'] == 'black':
                    res.append(1)
                else:
                    res.append(0)

        str1 = "0"
        for k in res:
            str1 += "," + str(k)
        f = open("tmp.csv", "w+")
        f.write(str1)
        f.close()

    def dump_data_test(self):
        res = []
        for i in self._widgets:
            for j in i:
                if j['bg'] == 'black':
                    res.append(1)
                else:
                    res.append(0)

        str1 = ""+self.example[self.parent.combo1.get()]
        for k in res:
            str1 += "," + str(k)
        str1 += "\n"
        f = open("test.csv", "a+")
        f.write(str1)
        f.close()

    def dump_data(self):
        res = []
        for i in self._widgets:
            for j in i:
                if j['bg'] == 'black':
                    res.append(1)
                else:
                    res.append(0)

        str1 = ""+self.example[self.parent.combo1.get()]
        for k in res:
            str1 += "," + str(k)
        str1 += "\n"
        f = open("train.csv", "a+")
        f.write(str1)
        f.close()


if __name__ == "__main__":
    lab = ExampleApp()

    try:
        with open('saved.data', 'rb') as f:
            svazi = pickle.load(f)
        lab.input_hidden_w = svazi[0]
        lab.hidden_out_w = svazi[1]
        lab.mainloop()
    except:
        lab.mainloop()
    res = [lab.input_hidden_w, lab.hidden_out_w]
    with open("saved.data", "wb") as f:
        pickle.dump(res, f)
