from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import numpy as np
import json
import os
import re
from dash import Dash, html, dcc, ctx
import dash_daq as daq
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State

import logging as log
from random import choices


# from server import launch_server

# Configure logging
log.basicConfig(level=log.INFO, format='%(levelname)s - %(message)s')


base_stylesheet_DTMC = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)', # Ensure labels are displayed
            'text-valign': 'center',
            'color': 'white',
            'text-outline-width': 1,
            'text-outline-color': 'black',
            'content': 'data(label)', # Display node text
            'text-wrap': 'wrap',
            'font-size': '20px',  # Set the font size to 20 pixels
            'text-max-width': 180,
            'width': '40px',
            'height': '40px',
        }
    },{
        'selector': 'edge',
        'style': {
            'label': 'data(p)',  # Ensure labels are displayed
            'text-valign': 'top',  # Adjust the position of the label
            'color': 'black',  # Set the label color
            'font-size': '16px',  # Adjust the font size of the label
            'text-opacity': 0.8,  # Adjust the opacity of the label
            'text-outline-color': 'white',  # Add an outline to the label text
            'text-outline-width': 0.5,
            'text-background-opacity': 1,  # Adjust the background opacity of the label
            'text-background-color': 'white',  # Set the background color of the label
            'text-background-padding': '2px',  # Adjust the padding of the label background
            'curve-style': 'unbundled-bezier',  # Set the curve style of the edge
            'target-arrow-shape': 'triangle',  # Add arrow to the target (destination) end
            'target-arrow-color': 'black',  # Set the color of the target arrow
            'line-color': 'black',  # Set the color of the edge line
            'width': 2,  # Adjust the width of the edge line
            'control-point-step-size': 80  # Adjust the distance between control points for more curvature
        }
    }
]

base_stylesheet_MDP = [
    {
        'selector': 'node[type="state"]',
        'style': {
            'label': 'data(label)',
            'text-valign': 'center',
            'color': 'white',
            'background-color': 'white',
            'border-color': 'black',
            'border-width': 1,
            'text-outline-color': 'black',
            'text-outline-width': 1,
            'content': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': 180,
            'width': '40px',
            'height': '40px',
        }
    },
    {
        'selector': 'node[type="action"]',
        'style': {
            'label': 'data(label)',
            'text-valign': 'center',
            'color': 'black',
            'background-color': 'black',
            'text-outline-color': 'white',
            'text-outline-width': 1,
            'content': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': 180,
            'width': '40px',
            'height': '40px',
        }
    },{
        'selector': 'edge',
        'style': {
            'label': 'data(p)',  # Ensure labels are displayed
            'text-valign': 'top',  # Adjust the position of the label
            'color': 'black',  # Set the label color
            'font-size': '16px',  # Adjust the font size of the label
            'text-opacity': 0.8,  # Adjust the opacity of the label
            'text-outline-color': 'white',  # Add an outline to the label text
            'text-outline-width': 0.5,
            'text-background-opacity': 1,  # Adjust the background opacity of the label
            'text-background-color': 'white',  # Set the background color of the label
            'text-background-padding': '2px',  # Adjust the padding of the label background
            'curve-style': 'unbundled-bezier',  # Set the curve style of the edge
            'target-arrow-shape': 'triangle',  # Add arrow to the target (destination) end
            'target-arrow-color': 'black',  # Set the color of the target arrow
            'line-color': 'black',  # Set the color of the edge line
            'width': 2,  # Adjust the width of the edge line
            'control-point-step-size': 80  # Adjust the distance between control points for more curvature
        }
    }
]



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

    ### Start of DEPRECATED functions 
    def generate_table(self):
        """
            Returns a table allowing each state to have an unique index
            !DEPRECATED
        """
        table = {}
        for i in range(len(self.states)):
            table[self.states[i]] = i
        return table

    def generate_matrix_DTMC(self):
        """
            Returns the transition matrix with the corresponding weights of the DTMC
            !DEPRECATED
        """
        n = len(self.states)
        table = self.generate_table()
        res = np.zeros(shape=(n, n))
        for t in self.transnoact:
            i = table[t.src]
            j = table[t.dst]
            res[i][j] = t.weight/t.total_weight
        return res

    def next_iteration(self, probas):
        """
            Returns the index in the table (same as the states property) of the resulting state
            !DEPRECATED    
        """

        p = np.random.rand()
        print(f"\nRNG: {p}")
        x = 0
        N = len(probas)
        for i in range(N):
            if p <= x+probas[i]:
                print(f"choice is {i}")
                return i
            else:
                x += probas[i]
        print(f"choice is {N-1}")
        return N-1 # Ensure that even with an unfortunate rounding, it returns a value
    ### End of DEPRECATED functions

    def __init__(self):
        self.states = []
        self.actions = []
        self.transact = []
        self.transnoact = []
        
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
        #TODO check only one occurence of each dst for a given scr
        #TODO Check that no action is named like a state
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

    def define_total_weights(self):
        is_DTMC = self.is_DTMC()

        #TODO Prettify this function

        for s in self.states:
            
            to_edit = [t for t in self.transnoact if t.src==s]
            total_weight = sum([t.weight for t in to_edit])
            for t in to_edit:
                t.total_weight = total_weight
            if not is_DTMC:
                for a in self.actions:
                    to_edit = [t for t in self.transact if t.src==s and t.action==a]
                    total_weight = sum([t.weight for t in to_edit])
                    for t in to_edit:
                        t.total_weight = total_weight

    def choose_iteration(self, transitions):
        """Returns the resulting state id based on the possible transitions"""
        #TODO Handle the case of no next transition, STOP the process ?
        weights = [t.weight for t in transitions]
        dests = [t.dst for t in transitions]
        print(f"\tWeights are {weights} for {dests}")
        p = choices(transitions, weights=weights)
        # print(p[0].dst)
        return p[0].dst

    def next_state(self, curr_state_id, action):
        """Returns the id of the next state based on the current state and the chosen action if one"""
        if action:
            trans = [t for t in self.transact if t.src==curr_state_id and t.action==action] 
        else:
            trans = [t for t in self.transnoact if t.src==curr_state_id]
        print("\n\t - Computing next state")
        if trans==[]:
            print(f"\tNo transition from {curr_state_id}, staying here")
            return curr_state_id
        print(f"\tCurrent state is {curr_state_id}")
        next_state_id = self.choose_iteration(trans)
        print(f"\tNext state is {next_state_id}")
        print("\t - End of next state computing\n")

        return next_state_id

    #TODO add a legend for DTMC and for MDP graphs

    def are_next_nodes_actions(self, curr_state_id):
        if curr_state_id:
            for t in self.transact:
                if t.src == curr_state_id:
                    return True
        return False

    def launch_server_DTMC(self):
        log.info("Launching server")
        app = Dash(__name__)

        nodes = [ {'data': {'id': s, 'label': s}} for s in self.states]
        edges = [ {'data': {'source': t.src, 'target': t.dst, 'p': f"{t.weight}/{t.total_weight}"}} for t in self.transnoact]
        # edges = [ {'data': {'source': t.src, 'target': t.dst, 'p': round(t.weight/t.total_weight, 2)}} for t in self.transnoact]
        starting_node_id = self.states[0]
        time_interval = 10

        app.layout = html.Div([
            html.P("Dash Cytoscape:"),
            html.Div(id='iteration-counter'),
            cyto.Cytoscape(
                id='cytoscape',
                elements=nodes + edges,
                layout={'name': 'circle'}, #'random' or 'circle'
                style={'width': '900px', 'height': '800px'}, #TODO Ensure a nice display
            ),
            # html.Div(id='state-info'),
            # dcc.Store(id='iteration-counter', data=0), # Store the iteration number
            dcc.Interval(
                id='interval',
                interval=time_interval*1000, # in milliseconds
                n_intervals=0,
            )
        ])
            

        def update_stylesheet(previous_node_id, current_node_id):
            stylesheet = base_stylesheet_DTMC.copy()
            log.info("\n  in update_stylesheet")
            if current_node_id:
                print(f"\tnode id is {current_node_id}")
                stylesheet.append({
                    'selector': f'node[id="{current_node_id}"]',
                    'style': {
                        'background-color': 'red',
                        'width': '50px',
                        'height': '50px',
                        'font-size': '28px',
                    }
                })
                stylesheet.append({
                        'selector': f'edge[source="{current_node_id}"]',
                        'style': {
                            'color': 'red',
                            'target-arrow-color': 'red',
                            'line-color': 'red',
                        }
                })
                if previous_node_id:
                    print(f"\tprevious edge is {previous_node_id} to {current_node_id}")
                    stylesheet.append({
                        'selector': f'edge[source="{previous_node_id}"][target="{current_node_id}"]',
                        'style': {
                            'color': 'blue',
                            'target-arrow-color': 'blue',
                            'line-color': 'blue',
                        }
                    })
            return stylesheet


        @app.callback(
            [Output('cytoscape', 'stylesheet'), Output('iteration-counter', 'children')],
            [Input('interval', 'n_intervals'), Input('cytoscape', 'stylesheet')],
            prevent_initial_call=False,
        )
        def next_node(n_intervals, stylesheet):
            log.info("\n  in next_node")
            iteration_str = f"Currently at iteration {n_intervals}"
            print(iteration_str)
            if stylesheet:
                for style in stylesheet:
                    if 'style' in style and 'background-color' in style['style']: # Get the value of the current node i.e. with a modified background
                        curr_node_id = style['selector'][9:-2]
                        next_node_id = self.next_state(curr_state_id=curr_node_id, action=None)
                        print(f"Next node id is {next_node_id}, waiting {time_interval}s\n")
                        return update_stylesheet(curr_node_id, next_node_id), iteration_str
            else:
                print("Only for init")
                return update_stylesheet(None, starting_node_id), iteration_str

        app.run_server(debug=True, use_reloader=False)




    def launch_server_MDP(self):
        log.info("Launching server")
        app = Dash(__name__)
        states = [{'data': {'id': s, 'label': s, 'type': 'state'}} for s in self.states]
        # actions = [{'data': {'id': a,'label': a,'type': 'action'}} for a in self.actions] #!DECREPATED
        actions = [{'data': {'id': f'{t.src}:{t.action}','label': t.action,'type': 'action'}} for t in self.transact ]
        nodes = states + actions
        for n in nodes:
            print(n)

        transnoact = [{'data': {'source': t.src,'target': t.dst,'p': f"{t.weight}/{t.total_weight}"}} for t in self.transnoact]
        # transact = [{'data': {'source': t.src,'target': t.dst,'p': f"{t.weight}/{t.total_weight}",'action': t.action}} for t in self.transact]
        pre_actions = [{'data': {'source': t.src,'target': f'{t.src}:{t.action}'}} for t in self.transact]
        post_actions = [{'data': {'source': f'{t.src}:{t.action}','target': t.dst, 'p': f"{t.weight}/{t.total_weight}"}} for t in self.transact]
        edges = transnoact + pre_actions + post_actions #TODO Colour the 2 types of action related edges ?
        for e in edges:
            print(e)

        starting_node_id = self.states[0] # Initial state is the first one by convention
        time_interval = 3

        app.layout = html.Div([
            html.P("Dash Cytoscape:"),
            html.Div([
                html.Div(id='iteration-counter'),
                html.Div(id='status'),
                daq.ToggleSwitch(
                    id='toggle-switch',
                    value=True,
                    label='Autonomous route',
                    labelPosition='bottom',
                    # vertical=True,
                    color='grey',
                ), #TODO Improve layout of this button
            ]),
            cyto.Cytoscape(
                id='cytoscape',
                elements=nodes + edges,
                layout={'name': 'grid'}, #'random' or 'circle' #TODO Search a better layout
                style={'width': '720px', 'height': '720px'}, #TODO Ensure a nice display based on % of the screen rather than px?
            ),
            dcc.Interval(
                id='interval',
                interval=time_interval*1000, # in milliseconds
                n_intervals=0,
            ),
        ])
            
        def update_stylesheet(previous_node_id, current_node_id):
            #TODO Unify this and the DTMC version ?
            stylesheet = base_stylesheet_MDP.copy()
            # log.info("\n  in update_stylesheet")
            if current_node_id:
                # print(f"    current node id is {current_node_id}")
                stylesheet.append({
                    'selector': f'node[id="{current_node_id}"]',
                    'style': {
                        'background-color': 'red',
                        'border-color': 'red',
                        'border-width': 1,
                        'width': '50px',
                        'height': '50px',
                    }
                })
                stylesheet.append({
                        'selector': f'edge[source="{current_node_id}"]',
                        'style': {
                            'color': 'red',
                            'target-arrow-color': 'red',
                            'line-color': 'red',
                        }
                })
                if previous_node_id:
                    print(f"\tprevious edge is {previous_node_id} to {current_node_id}")
                    stylesheet.append({
                        'selector': f'edge[source="{previous_node_id}"][target="{current_node_id}"]',
                        'style': {
                            'color': 'blue',
                            'target-arrow-color': 'blue',
                            'line-color': 'blue',
                        }
                    })
                    #TODO make bigger previous and next edges ?
            print()
            return stylesheet


        @app.callback(
            [
                Output('cytoscape', 'stylesheet'),
                Output('iteration-counter', 'children'),
                Output('status', 'children'),
                Output("interval", "disabled"),
            ],[
                Input('interval', 'n_intervals'),
                Input('cytoscape', 'tapNodeData'),
                Input('cytoscape', 'stylesheet'),
                Input('toggle-switch', 'value'),
            ],
            prevent_initial_call=False,
        )
        def next_node(n_intervals, tapped_node, stylesheet, switch_value):

            log.info("### Choosing next node ###")
            iteration_str = f"Currently at iteration {n_intervals}"
            if tapped_node:
                print(f"\ttapped node is {tapped_node}")
            else:
                print(f"\tno node tapped")

            to_disable = False
            #TODO Add a button to enable/disable the timer 

            # return update_stylesheet(None, starting_node_id), iteration_str
            # print(iteration_str)
            curr_node_id = None
            prev_node_id = None
            if stylesheet:
                for style in stylesheet:
                    # print(style)
                    if not prev_node_id and style['style'].get('line-color')=='blue': # Get the value of the previous node i.e. with a blue background
                        pattern = r'edge\[source="(\w+:?\w+)"\]\[target="\w+:?\w+"\]'
                        match = re.search(pattern, style['selector'])
                        if match:
                            prev_node_id = match.group(1)
                            print(f"\tprev_node is {prev_node_id}")
                    if not curr_node_id and style['style'].get('background-color')=='red': # Get the value of the current node i.e. with a red background
                        curr_node_id = style['selector'][9:-2]
                        print(f"\tcurr_node is {curr_node_id}")

            if not curr_node_id: # No node selected
                print("--> Graph initialized")
                return update_stylesheet(None, starting_node_id), iteration_str, 'Graph initialized', False #?

            #TODO Add comments 
            #TODO Clean code
            # print(f"\tactions ? for node {curr_node_id}: {self.are_next_nodes_actions(curr_node_id)}")
            # print(f"\t{tapped_node and tapped_node.get('type', '')=='action'}")

            if ctx.triggered_id and ctx.triggered_id=='toggle-switch':
                print(f"Autonomous route is set to {switch_value}")
                return update_stylesheet(prev_node_id, curr_node_id), iteration_str, '\n', not switch_value


            if self.are_next_nodes_actions(prev_node_id):
                print("\tCurrently in an action")
                source, action = curr_node_id.split(':') # Current node is an action thus in 'source:action' format
                next_node_id = self.next_state(source, action) 
                return update_stylesheet(curr_node_id, next_node_id), iteration_str, '\n', self.are_next_nodes_actions(next_node_id)

            elif self.are_next_nodes_actions(curr_node_id): # We are in a action choosing step and waiting for a tap on an action node
                if tapped_node and tapped_node.get('type', '')=='action': # We tapped an action
                    action = tapped_node.get('id', None)
                    print(f"\taction is {action}")

                    source = action.split(':')[0]
                    print(f"RES {source} {curr_node_id}")
                    if source != curr_node_id: #TODO Refactor this
                        print('\twrong action tapped')
                        return update_stylesheet(prev_node_id, curr_node_id), iteration_str, 'Wrong action tapped', True

        #TODO Improve status text zone

                    if action:
                        return update_stylesheet(curr_node_id, action), '', iteration_str, False
                        # next_node_id = self.next_state(curr_state_id=curr_node_id, action=action)
                        # print(f"    Chosen action {action}, next node id is {next_node_id}, waiting {time_interval}s")
                        # return update_stylesheet(curr_node_id, next_node_id), iteration_str
                else:
                    print('\taction needed but tapped node is not')
                    return update_stylesheet(prev_node_id, curr_node_id), iteration_str, 'Tapped node is not an action', True

            else:
                next_node_id = self.next_state(curr_state_id=curr_node_id, action=None)
                print(f"\tNext node id is {next_node_id}, waiting {time_interval}s")
                return update_stylesheet(curr_node_id, next_node_id), iteration_str, '\n', self.are_next_nodes_actions(next_node_id)



                next_node_id = curr_node_id
                print(f"    Next node id is {next_node_id}, waiting {time_interval}s\n")
                return update_stylesheet(curr_node_id, next_node_id), iteration_str
            
    

        # @app.callback(
        #         [Output('cytoscape', 'stylesheet'), Output('iteration-counter', 'children')],
        #         [Input('interval', 'n_intervals'), Input('cytoscape', 'tapNodeData'), Input('cytoscape', 'stylesheet')],
        #         prevent_initial_call=False,
        # )
        # def select_action(self, current_node_id, stylesheet):
            
        #     action = 0
        #     return action





        app.run_server(debug=True, use_reloader=False)








def main():

    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    # print('Before\n')
    walker.walk(printer, tree)
    # print('\nAfter\n')
    if printer.check_validity():
        printer.define_total_weights()
        if printer.is_DTMC():
            print("Graph type is DTMC\n")
            printer.launch_server_DTMC()


        else:
            print("Graph type is MDP")
            printer.launch_server_MDP()
            #TODO Implement for the MDP



if __name__ == "__main__":
    print('\n#####\nin the call\n#####\n')
    main()


# Code to run in the cli is :
# cmd /c 'python mdp.py < DTMC.mdp'