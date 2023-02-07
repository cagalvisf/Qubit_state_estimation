## Graphical comparison between estimation in HELMI and IBMQ
## Imports
import sys
import argparse
from argparse import RawTextHelpFormatter

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, BasicAer, execute
from qiskit.compiler import transpile
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import plot_bloch_vector
from qiskit.quantum_info import state_fidelity
from qiskit.transpiler.passes import RemoveBarriers
from qiskit import IBMQ

from qubit_state_estimation.interaction import IBMQ_U, U_matrix, T_matrix
from qubit_state_estimation.estimators import linear_estimation, disc_ML_estimation

import matplotlib.pyplot as plt
import numpy as np

pi = np.pi
j = complex(0,1)
shots = 1024
def get_args():

    args_parser = argparse.ArgumentParser(
        description="""Qubit state estimation for Pauli eigenstates""",
        formatter_class=RawTextHelpFormatter,
        epilog="""Example usage:
        python Complete_QSE.py --backend simulator
        """,
    )

    # Parse Arguments

    args_parser.add_argument(
        "--backend",
        help="""
        Define the backend for running the program.
        'simulator' runs on Qiskit's aer simulator,
        'IBMQ' runs on an IBM QPU,
        'helmi' runs on VTT Helmi Quantum Computer,
        'fake_helmi' runs in the fake helmi simulator
        """,
        required=False,
        type=str,
        default=None,
        choices=["helmi", "simulator", "IBMQ", "fake_helmi"],
    )


    return args_parser.parse_args()

def Initialization(angles = [0,0,0]):

    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)

    #Initialice qubits
    qc.h(qr[0])                                    #A meter
    qc.h(qr[2])                                    #B meter
    qc.u(angles[0],angles[1],angles[2],[qr[1]])    #System

    Init_gate = qc.to_gate()
    Init_gate.name= "Initialization"

    return Init_gate

def tg_circuit(angles_i):
    ## We define the tomography circuit as a function
    A = QuantumRegister(1,'a')
    S = QuantumRegister(1,'s')
    B = QuantumRegister(1,'b')
    cr = ClassicalRegister(3)
    
    ## We use the best values of the angles for A and B
    th_a1,phi_a1,lam_a1,th_b1,phi_b1,lam_b1 = 0.5871626,  1.57737493, 2.52063619, 0.70004151, 4.30553732, 3.45993977
    th_a2,phi_a2,lam_a2,th_b2,phi_b2,lam_b2 = 2.55283129, 1.93819982, 0.30976956, 0.67288561, 6.47455126, 4.4695403
    
    angles_a=[th_a1,phi_a1,lam_a1,th_a2,phi_a2,lam_a2]
    angles_b=[th_b1,phi_b1,lam_b1,th_b2,phi_b2,lam_b2]
    
    ## Creation of the circuit
    tomography_circuit = QuantumCircuit(A,S,B, cr)

    ## Initialize the ASB qubits
    tomography_circuit.append(Initialization(angles_i),[A,S,B])

    ## Apply the evolution operator
    tomography_circuit.append(IBMQ_U(angles_a,angles_b),[A,S,B])

    ## Change of the measurement basis
    tomography_circuit.h(A)                     #A
    tomography_circuit.h(B)                     #B

    ## Measure
    tomography_circuit.measure(A,cr[0])         #A
    tomography_circuit.measure(B,cr[2])         #B
    
    return tomography_circuit

def bloch_vector_to_state(s):
    ## We transform the bloch vector into a qubit state
    s1 = s[0]
    s2 = s[1]
    s3 = s[2]

    if (s3>0):
        th=np.arctan(np.sqrt(s1*s1 + s2*s2)/s3)
    elif (s3 == 0):
        th = pi/2
    elif (s3 < 0):
        th=np.arctan(np.sqrt(s1*s1 + s2*s2)/s3) + pi

    if (s1>0 and s2>=0):
        phi = np.arctan(s2/s1)
    elif (s1>0 and s2<0):
        phi = np.arctan(s2/s1) + 2*pi
    elif (s1 == 0):
        phi = pi/2*np.sign(s2)
    elif (s1<0):
        phi = np.arctan(s2/s1) + pi

    ## We write the alpha and beta amplitudes
    c_0 = np.cos(th/2)
    c_1 = (np.cos(phi) + j*np.sin(phi))*np.sin(th/2)

    state = np.array([c_0,c_1])
    return state

def init_ang_to_bloch_vector(angles):

    th, phi, lam = angles

    s = np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)])

    return np.round(s,2)

def simulation(angles_i, device="simulator"):
    tomography_circuit = tg_circuit(angles_i)

    print("Running on backend = ", device)

    if device == "simulator" or device == "aer":

        backend = Aer.get_backend('qasm_simulator')
        basis_gates = backend.configuration().basis_gates
        tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)

        tomography_circuit = RemoveBarriers()(tomography_circuit)
        sim_result_tomography_circuit = backend.run(tomography_circuit, shots=shots).result()
    
    elif device == "IBMQ":
        IBMQ.load_account()

        provider=IBMQ.get_provider('ibm-q')
        backend = provider.get_backend('ibmq_lima')

        basis_gates = backend.configuration().basis_gates
        tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)
        tomography_circuit = RemoveBarriers()(tomography_circuit)

        print("Running on ", backend.configuration().backend_name)

        sim_result_tomography_circuit = backend.run(tomography_circuit, shots=shots).result()

    elif device == "helmi":
        from csc_qu_tools.qiskit import Helmi as helmi
        provider = helmi()
        backend = provider.set_backend()
        basis_gates = provider.basis_gates

        tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)
        tomography_circuit = RemoveBarriers()(tomography_circuit)

        virtual_qubits = tomography_circuit.qubits
        qubit_mapping = {
                virtual_qubits[0]: "QB1",
                virtual_qubits[1]: "QB3",
                virtual_qubits[2]: "QB2",
            }
        sim_result_tomography_circuit = backend.run(tomography_circuit, shots=shots, qubit_mapping=qubit_mapping).result()

    elif device == "fake_helmi":
        from csc_qu_tools.qiskit.mock import FakeHelmi

        print(
                "Inducing artificial noise into Simulator with FakeHelmi Noise Model"
            )
        basis_gates = ["r", "cz"]
        backend = FakeHelmi()

        tomography_circuit = transpile(tomography_circuit, basis_gates=basis_gates)
        tomography_circuit = RemoveBarriers()(tomography_circuit)

        virtual_qubits = tomography_circuit.qubits
        qubit_mapping = {
                virtual_qubits[0]: "QB1",
                virtual_qubits[1]: "QB3",
                virtual_qubits[2]: "QB2",
            }
        sim_result_tomography_circuit = backend.run(tomography_circuit, shots=shots, qubit_mapping=qubit_mapping).result()


    else:
        sys.exit("Backend option not recognized")

    sim_counts = sim_result_tomography_circuit.get_counts()
  
    
    if '000' in sim_counts:
        p00 = sim_counts['000']/shots
    else:
        p00 = 0

    if '001' in sim_counts:
        p01 = sim_counts['001']/shots
    else:
        p01 = 0
    if '100' in sim_counts:
        p10 = sim_counts['100']/shots
    else:
        p10 = 0
    if '101' in sim_counts:
        p11 = sim_counts['101']/shots
    else:
        p11 = 0


    p = np.array([p00,p01,p10,p11])

    return p


## Number of estimations for each state
N=5
args = get_args()
device = args.backend

th_a1,phi_a1,lam_a1,th_b1,phi_b1,lam_b1 = 0.5871626,  1.57737493, 2.52063619, 0.70004151, 4.30553732, 3.45993977
th_a2,phi_a2,lam_a2,th_b2,phi_b2,lam_b2 = 2.55283129, 1.93819982, 0.30976956, 0.67288561, 6.47455126, 4.4695403

angles_a=[th_a1,phi_a1,lam_a1,th_a2,phi_a2,lam_a2]
angles_b=[th_b1,phi_b1,lam_b1,th_b2,phi_b2,lam_b2]

U = U_matrix(angles_a,angles_b)
T = T_matrix(U)
## File where the results will be stored
file = open("data/"+args.backend+"_data.txt","a")

## Initial angles for Pauli matrices eigenstates 
ang_z0 = [0,0,0]
ang_z1 = [pi,0,0]
ang_x0 = [pi/2,0,0]
ang_x1 = [-pi/2,0,0]
ang_y0 = [-pi/2,-pi/2,pi/2]
ang_y1 = [pi/2,-pi/2,pi/2]


angles = [ang_z0,ang_z1,ang_x0,ang_x1,ang_y0,ang_y1]
states = ['z0','z1','x0','x1','y0','y1']

state = 0
for a_i in angles:
    
    for n in range(N):
        
        p_IBM = simulation(a_i, device=device)

        s_real = init_ang_to_bloch_vector(a_i)
        ## Estimation of the state, can be "linear" or "ml"
        s_l_est  = linear_estimation(T, p_IBM)
        s_ml_est = disc_ML_estimation(T, p_IBM)

        state_ideal  = bloch_vector_to_state(s_real)
        state_l_est  = bloch_vector_to_state(s_l_est)
        state_ml_est = bloch_vector_to_state(s_ml_est)
        
        fidelity_l  = state_fidelity(state_ideal, state_l_est)
        fidelity_ml = state_fidelity(state_ideal, state_ml_est)
        
        print('Probabilities: ', p_IBM)
        print('s ideal: ', s_real)
        print('s linear: ', s_l_est)
        print('s ML: ', s_ml_est)

        print('Fidelity linear estimation: ',fidelity_l)
        print('Fidelity ML estimation: ',fidelity_ml)
        ## Columns: State, P_00, P_01, P_10, P_11, s_x, s_y, s_z, se_l_x, se_l_y, se_l_z, fidelity_l, se_ml_x, se_ml_y, se_ml_z, fidelity_ml
        file.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s  \n" % 
                (states[state], p_IBM[0], p_IBM[1],p_IBM[2], p_IBM[3], s_real[0],s_real[1],s_real[2], s_l_est[0],s_l_est[1],s_l_est[2], fidelity_l, s_ml_est[0], s_ml_est[1], s_ml_est[2], fidelity_ml))
        
        print('State: ', states[state], 'run: ',n+1,' Finished.')
    state += 1
file.close()
