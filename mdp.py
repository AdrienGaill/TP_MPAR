from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import numpy as np

from dash import Dash, html, no_update
import dash_cytoscape as cyto
from dash.dependencies import Input, Output

import logging as log
from time import sleep
import threading



# from server import launch_server

# Configure logging
log.basicConfig(level=log.INFO, format='%(levelname)s - %(message)s')

class Transition():

    def __init__(self, src: str, dst: str, weight: int, action: str) -> None:
        self.src = src
        self.dst = dst
        self.weight = weight
        self.total_weight = 0
        self.action = action
    
    def _valid_src(self, states):
        return self.src in states

    def _valid_dst(self, states):
        return self.dst in states

    def _valid_action(self, actions):
        # print(self.transition_to_str())
        if self.action is not None:
            return self.action in actions
        return True

    def _valid_weight(self):
        return isinstance(self.weight, int) and self.weight>0 #and self.weight<=self.total_weight
        
    def is_valid(self, states, actions):
        return (
            self._valid_src(states) and
            self._valid_dst(states) and
            self._valid_action(actions) and
            self._valid_weight()
        )
    
    def transition_to_str(self):
        return f"Transition from {self.src} to {self.dst} by action {self.action} with weight {self.weight} on {self.total_weight}"


class gramPrintListener(gramListener):

    def __init__(self):
        self.states = []
        self.actions = []
        self.transact = []
        self.transnoact = []
        self.app = None
        
    def enterDefstates(self, ctx):
        states = [str(x) for x in ctx.ID()]
        self.states = states
        print(f"States: {states}")


    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.actions = actions
        print(f"Actions: {actions}")

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        weights = [int(str(x)) for x in ctx.INT()]
        total_weight = sum(weights)
        # print("Total weight is", total_weight)
        src = ids.pop(0)
        act = ids.pop(0)
        while weights:
            t = Transition(
                src=src,
                dst=ids.pop(0),
                weight=weights.pop(0),
                action=act,
            )
            if t.is_valid(self.states, self.actions):
                self.transact.append(t)
                # print(f"transact from {t.src} to {t.dst} by action {t.action} with weight {t.weight}")
            else:
                print(f"transact from {t.src} to {t.dst} by action {t.action} with weight {t.weight} not added")

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        weights = [int(str(x)) for x in ctx.INT()]
        src = ids.pop(0)
        while weights:
            t = Transition(
                src=src,
                dst=ids.pop(0),
                weight=weights.pop(0),
                action=None,
            )
            if t.is_valid(self.states, self.actions):
                self.transnoact.append(t)
                # print(f"transact from {t.src} to {t.dst} by action {t.action} with weight {t.weight}")
            else:
                print(f"transnoact from {t.src} to {t.dst} by action {t.action} with weight {t.weight} not added")

    def check_validity(self):
        is_valid = True
        print("Checking debut")

        for s in self.states:
            with_action, without_action = False, False

            print(f"Started checking {s}")
            src_transact = [t for t in self.transact if t.src==s]
            for t in src_transact:
                with_action = True
                print(f"    Checked from {t.src} to {t.dst} by action {t.action} with weight {t.weight}")
                #TODO All checks here
            src_transnoact = [t for t in self.transnoact if t.src==s]
            for t in src_transnoact:
                without_action = True
                #TODO All checks here
                print(f"    Checked from {t.src} to {t.dst} by action {t.action} with weight {t.weight}")

            if with_action and without_action:
                print(f"    Both action and no_action for transition state [{s}] -> Error")
                is_valid = False

            print(f"Finished checking {s}")
        
        print(f"\nChecking end: result is {is_valid}\n")
        return is_valid

    def is_DTMC(self):
        return len(self.transact)==0

    def generate_table(self):
        table = {}
        for i in range(len(self.states)):
            table[self.states[i]] = i
        return table

    def define_total_weights(self):
        is_DTMC = self.is_DTMC()
        for s in self.states:
            if is_DTMC:
                to_edit = [t for t in self.transnoact if t.src==s]
            else:
                to_edit = [t for t in self.transact if t.src==s]

            total_weight = sum([t.weight for t in to_edit])
            for t in to_edit:
                t.total_weight = total_weight

    def generate_matrix_DTMC(self):
        n = len(self.states)
        table = self.generate_table()
        res = np.zeros(shape=(n, n))
        for t in self.transnoact:
            i = table[t.src]
            j = table[t.dst]
            res[i][j] = t.weight/t.total_weight

        # print(res)
        return res

    def iterate_over_DTMC(self, curr_state):
        # print(f"Starting from {curr_state}")
        table = self.generate_table()
        probas = self.generate_matrix_DTMC()[table[curr_state]]
        next_state = self.states[self.next_iteration(probas)]
        # print(next_state)
        return next_state

    def next_iteration(self, probas):
        p = np.random.rand()
        print(f"RNG: {p}")
        x = 0
        for i in range(len(probas)):
            if p <= x+probas[i]:
                # print(f"choice is {i}")
                return i
            else:
                x += probas[i]

    def run_DTMC(self):
        curr_state = self.states[0]
        while True:
            curr_state =  self.iterate_over_DTMC(curr_state)
            print(curr_state)
            sleep(5)
            # self.simulate_click(curr_state)


    def simulate_click(self, node_id):
        # Get the callback associated with the output
        callback = self.app.callback_map.get('cytoscape.stylesheet', {}).get('callback')

        # Ensure the callback exists
        if callback:
            # Call the callback function directly with the input value
            callback(node_id, outputs_list=[{'id': 'cytoscape', 'property': 'stylesheet'}])














    def launch_server(self):
        log.info("Launching server")
        app = Dash(__name__)

        nodes = [ {'data': {'id': s, 'label': s}} for s in self.states]
        edges = [ {'data': {'source': t.src, 'target': t.dst}} for t in self.transnoact]

        app.layout = html.Div([
            html.P("Dash Cytoscape:"),
            cyto.Cytoscape(
                id='cytoscape',
                elements=nodes + edges,
                layout={'name': 'circle'},
                style={'width': '720px', 'height': '540px'}
            ),
            html.Div(id='node-info')
        ])

        @app.callback(
            Output('node-info', 'children'),
            [Input('cytoscape', 'tapNodeData')]
        )
        def display_node_info(node_data):
            if node_data:
                return f"You clicked on node: {node_data['label']}"
            else:
                return ""
                    
        @app.callback(
            Output('cytoscape', 'stylesheet'),
            [Input('cytoscape', 'tapNodeData')]
        )
        def update_stylesheet(node_data):
            stylesheet = [{
                'selector': 'node',
                'style': {
                    'label': 'data(label)',  # Ensure labels are displayed
                    'text-valign': 'center',
                    'color': 'white',
                    'text-outline-width': 2,
                    'text-outline-color': 'green',
                    'content': 'data(label)',  # Display node text
                    'text-wrap': 'wrap',
                    'text-max-width': 80
                }
            }]
            if node_data:
                stylesheet.append({
                    'selector': f'node[id="{node_data["id"]}"]',
                    'style': {
                        'background-color': 'red'
                    }
                })
            return stylesheet



        self.app = app
        app.run_server(debug=True, use_reloader=False)

        # self.run_DTMC()






#TOSELF Works correctly but need to implement a stochastic march over the graph




def main():
    log.warning('main is starting')

    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    print('Before\n')
    walker.walk(printer, tree)
    print('\nAfter\n')
    if printer.check_validity():
        printer.define_total_weights()
        if printer.is_DTMC():
            print("Graph type is DTMC\n")
            # mat = printer.generate_matrix_DTMC()
            # print('\n', mat, '\n')
            # printer.iterate_over_DTMC(printer.states[0])
            
            # server_thread = threading.Thread(target=printer.launch_server)
            # server_thread.start()
            
            printer.launch_server()

            # printer.run_DTMC()


            #TODO Visual representation

        else:
            print("MDP")
            #TODO Implement for the MDP

if __name__ == '__main__':
    print('\n#####\nin the call\n#####\n')
    main()


# Code to run in the cli is :
#   python mdp.py DTMC.mdp