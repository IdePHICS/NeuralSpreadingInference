import numpy as np
from scipy import special
from Modules.fg import *

epsilon = 2**(-200)

def generate_us(M, wRademacher=False):
    """
    Creates the vector of hidden variables. Gaussian or Rademacher.
    """
    if wRademacher:
        return np.random.randint(0, 2, M)*2-1
    else:
        return np.random.normal(0, 1, M)
    
def generate_F(N, M):
    """
    Creates the random features.
    """
    return np.random.normal(0,M**(-1/2),(N,M))

def compute_go2(nu, om, V, c):
    #g = np.array([-2/o if abs((o+c)*(2*V)**(-1/2)) > 5.9 else (2*np.pi*V)**(-1/2)*(2*nu[i] - 1)*np.exp(-(o+c)**2/(2*V))/((1+(2*nu[i] - 1)*special.erf((o+c)*(2*V)**(-1/2)))/2) for i,o in enumerate(om) ])
    Znn = (1+(2*nu-1)*special.erf((om+c)*(2*V)**(-1/2)))/2
    if (Znn == 0).sum() > 0:
        print('Znn = 0')
    Znn = np.maximum(epsilon, Znn)
    return (2*np.pi*V)**(-1/2)*(2*nu-1)*np.exp(-(om+c)**2/2/V)/Znn, Znn
    #return g, 0

def compute_go(nu, om, V, c):
    g = np.array([-(o+c)/V if ( (((o+c)*(2*V)**(-1/2) > 5.9) and (nu[i] < 1e-6)) or (((o+c)*(2*V)**(-1/2) < -5.9) and (nu[i] > 1 - 1e-6)) ) else (2*np.pi*V)**(-1/2)*(2*nu[i] - 1)*np.exp(-(o+c)**2/(2*V))/((1+(2*nu[i] - 1)*special.erf((o+c)*(2*V)**(-1/2)))/2) for i,o in enumerate(om) ])
    return g, 0

def fa(Lambda, Gamma, wRademacher=False):
    if wRademacher:
        return np.tanh(Gamma)
    else:
        return Gamma/(Lambda+1)

def fv(Lambda, Gamma, wRademacher=False):
    if wRademacher:
        return np.array([np.cosh(g)**(-2) if abs(g) < 100 else np.cosh(100)**(-2) for g in Gamma ])
        #return np.maximum(epsilon, np.cosh(Gamma)**(-2))
    else:
        return 1/(Lambda+1)

#qui metti la funzione per la free entropy glm 
def loglikelihood_AMP(f, F, a, g, v, A, B,c, wRademacher=False): #rademacher, c=0
    N = f.size
    M = F.shape[1]
    V = np.mean(v)
    Fa = np.dot(F, a) 
    omega = Fa - V * g 

    term1 = 0.0
    for alpha in range(M):
        #term1 += -A / 2.0 + np.log(np.cosh()) 
        if wRademacher: term1 += -A/2+np.logaddexp(B[alpha], -B[alpha])-np.log(2)
        else: term1 += -np.log(A+1)/2+B[alpha]**2/2/(A+1)
    term1 /= N

    term2 = 0.0
    for alpha in range(M):
        term2 += (A/ 2.0) * (a[alpha]**2 + v[alpha]) - B[alpha] * a[alpha]
    term2 /= N

    term3 = np.sum((V * g)**2) / (2 * V * N)

    nu_nn, norm = f.compute_nu()
    nu = nu_nn / norm 

    Znn = ((1+(2*nu-1)*special.erf((omega+c)*(2*V)**(-1/2)))/2 )
    Znn = np.maximum(epsilon, Znn)
    term4 = np.log(Znn).sum() / N

    return term1 + term2 + term3 + term4



def compute_onsager_terms(f, c):
    """
    Computes the Onsager parameters A and B for each latent variable according
    to the following formulas:
    
       A = (1/M) * sum_{i=1}^{N} [g_{o,i}]^2,
       B[α] = sum_{i=1}^{N} F[i, α] * g_{o,i} + a[α] * A,
       
    where:
       - f.g is an array of length N containing the output function g_{o,i},
       - f.F is the feature matrix of shape (N, M),
       - f.a is the current AMP estimate (vector of length M).
       
    The parameter c is not used in this computation but is kept in the signature
    for compatibility with the rest of the code.
    """
    N = f.size
    M = f.F.shape[1]
    A = np.sum(f.g**2) / M
    B = np.zeros(M)
    for alpha in range(M):
        B[alpha] = np.sum(f.F[:, alpha] * f.g) + f.a[alpha] * A
    f.A = A
    f.B = B


def compute_eta(om, V, c): #returns eta(- 1)
    #eta = np.zeros((len(om), 2))
    #eta[:,0] = (1+special.erf((c+om)*(2*V)**(-1/2)))/2
    #eta[:,1] = (1-special.erf((c+om)*(2*V)**(-1/2)))/2
    #eta = np.maximum(epsilon, eta)
    #return eta[:,1]*delta_0/(eta[:,1]*delta_0 + eta[:,0]*(1-delta_0))
    return (1-special.erf((c+om)*(2*V)**(-1/2)))/2#np.maximum(,epsilon)

def _go(chisP, om, V):
    Znn = (1+(2*chisP-1)*special.erf(om*(2*V)**(-1/2)))/2
    Znn = np.maximum(epsilon, Znn)
    return (2*np.pi*V)**(-1/2)*(2*chisP-1)*np.exp(-om**2/2/V)/Znn, Znn

def _go_ws(chisP, om, V):
    Znn = (1+(2*chisP-1)*special.erf(om*(2*V)**(-1/2)))/2
    Znn = np.maximum(epsilon, Znn)
    return (2*np.pi*V)**(-1/2)*(2*chisP-1)*np.exp(-om**2/2/V)/Znn, Znn


def _fa(Lambda, Gamma, wRademacher=False):
    if wRademacher:
        return np.tanh(Gamma)
    else:
        return Gamma/(Lambda+1)

def _fv(Lambda, Gamma, wRademacher=False):
    if wRademacher:
        return np.maximum(epsilon, np.cosh(Gamma)**(-2))
    else:
        return 1/(Lambda+1)


def stepAMP(a, v, chisP, goPrev, F, wRademacher=False):
    """
    Performs one step of AMP on the GLM side. Returns the updated variables and the free entropy.
    
    a: array M
    v: array M
    chisP: array N, namely chis[:,0]
    goPrev: array N
    F: array NxM
    """
    M, N = len(a), len(chisP)
    
    V = np.mean(v)
    om = np.dot(F, a) - V*goPrev
    
    #psis = _psis(om, V)
    go, Znn = _go(chisP, om, V)
    Lambda = np.sum(go**2)/M
    Gamma = a*Lambda+np.dot(go, F)
    
    #logZl = _logZl(Lambda, Gamma, wRademacher)
    #Fl = np.sum(logZl)+np.sum(Lambda*(v+a**2)/2-Gamma*a)+np.sum(go**2*V/2)
    #Fnn = np.sum(np.log(Znn))
    
    a = _fa(Lambda, Gamma, wRademacher)
    v = _fv(Lambda, Gamma, wRademacher)
    
    return a, v, go #psis, go, (Fl+Fnn)/N

def stepAMP_ws(a, v, chisP, goPrev, F, wRademacher=False):
    """
    Performs one step of AMP on the GLM side (without simplifications). Returns the updated variables and the free entropy.
    
    a: array M
    v: array M
    chisP: array N, namely chis[:,0]
    goPrev: array N
    F: array NxM
    """
    M, N = len(a), len(chisP)
    
    V = np.dot(F**2, v)#np.mean(v)
    om = np.dot(F, a) - V*goPrev
    
    #psis = _psis(om, V)
    go, Znn = _go_ws(chisP, om, V)
    Lambda = np.sum(go**2)/M
    Gamma = a*Lambda+np.dot(go, F)
    
    #logZl = _logZl(Lambda, Gamma, wRademacher)
    #Fl = np.sum(logZl)+np.sum(Lambda*(v+a**2)/2-Gamma*a)+np.sum(go**2*V/2)
    #Fnn = np.sum(np.log(Znn))
    
    a = _fa(Lambda, Gamma, wRademacher)
    v = _fv(Lambda, Gamma, wRademacher)
    
    return a, v, go #psis, go, (Fl+Fnn)/N


def AMPstep(damp,a_old, v_old, go_old,nu, F, c, wRademacher=False):
    """
    Performs one step of AMP. Returns the updated variables.
    
    a_old: array M
    v_old: array M
    go_old: array N
    F: array NxM
    c: float
    """ 
    M, N = len(a_old), len(go_old)
    V = np.mean(v_old)
    om = np.dot(F, a_old) - V*go_old
    #print(min(om), max(om), np.isnan(om).sum())
    #nu= (1-damp)*f.compute_nu() + damp*nu_old
    #nu= f.compute_nu()
    go, _ = compute_go(nu, om, V, c)
    go = (1-damp)*go +damp*go_old
    #print(go[:10],Znn[:10])
    #f.set_delta(eta)
    
    Av = np.sum(go**2)/M
    Bv = a_old*Av+np.dot(go, F)
    #print(min(go),max(go),np.isnan(go).sum())
    #print(Av)
    #print(min(Bv),max(Bv),np.isnan(Bv).sum())
    a_new = damp*a_old + (1-damp)*fa(Av, Bv, wRademacher)
    v_new = damp*v_old + (1-damp)*fv(Av, Bv, wRademacher)
    
    #err_max, err_mean = f.iterate(damp) #nu(+1)
    
    return  a_new, v_new, go

def stepAMP(a, v, chisP, goPrev, F, wRademacher=False):
    """
    Performs one step of AMP on the GLM side. Returns the updated variables and the free entropy.
    
    a: array M
    v: array M
    chisP: array N, namely chis[:,0]
    goPrev: array N
    F: array NxM
    """
    M, N = len(a), len(chisP)
    
    V = np.mean(v)
    om = np.dot(F, a) - V*goPrev
    
    #psis = _psis(om, V)
    go, Znn = _go(chisP, om, V)
    Lambda = np.sum(go**2)/M
    Gamma = a*Lambda+np.dot(go, F)
    
    #logZl = _logZl(Lambda, Gamma, wRademacher)
    #Fl = np.sum(logZl)+np.sum(Lambda*(v+a**2)/2-Gamma*a)+np.sum(go**2*V/2)
    #Fnn = np.sum(np.log(Znn))
    
    a = _fa(Lambda, Gamma, wRademacher)
    v = _fv(Lambda, Gamma, wRademacher)
    
    return a, v, go #psis, go, (Fl+Fnn)/N

def BPstep(f,damp):
    """
    Performs one step of BP. Returns the updated variables.
    
    a_old: array M
    v_old: array M
    go_old: array N
    F: array NxM
    c: float
    """ 
    
    return f.iterate(damp)

def BP_AMPstep(f,damp, a_old, v_old, go_old, eta_old, nu_old, F, c, wRademacher=False):
    """
    Performs one step of BP-AMP. Returns the updated variables.
    
    a_old: array M
    v_old: array M
    go_old: array N
    F: array NxM
    c: float
    """ 
    M, N = len(a_old), len(go_old)
    V = np.mean(v_old)
    om = np.dot(F, a_old) - V*go_old
    #print(min(om), max(om), np.isnan(om).sum())
    nu_nn, norm = f.compute_nu()
    nu = np.array([ (1-damp)*nu_nn[i]/norm[i] + damp*nu_old[i] if norm[i]>0 else nu_old[i]  for i in range(N)])
    #nu= f.compute_nu()
    go, _ = compute_go(nu, om, V, c)
    go = (1-damp)*go +damp*go_old
    #print("go"+f"{go[:10]}")
    eta = damp*eta_old + (1-damp)*compute_eta(om, V, c)
    f.set_delta(eta)
    #print("d"+f"{f.get_delta()[:10]}" + "om"+f"{om[:10]}"+ "V" + f"{V}" + "v_old" + f"{v_old[:10]}")
    #print("nu"+f"{nu[:10]}")
    Av = np.sum(go**2)/M
    Bv = a_old*Av+np.dot(go, F)
    #print(min(go),max(go),np.isnan(go).sum())
    #print("Bv: " + f"{np.dot(go, F)[:10]}" + "Av: " + f"{ (a_old*Av)[:10]}")
    #print(min(Bv),max(Bv),np.isnan(Bv).sum())
    a_new = damp*a_old + (1-damp)*fa(Av, Bv, wRademacher)
    v_new = damp*v_old + (1-damp)*fv(Av, Bv, wRademacher)
    
    _, [err_max, err_mean] = f.update(maxit=1000, tol=1e-6, damp=damp) #nu(+1)

    chi = np.array([1 if ((nu[i] == 1) or (eta[i]==0)) else 0 if ((eta[i]==1) or (nu[i] == 0)) else nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
    #chi = np.array([nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
    x_new = 2*chi-1
    
    return err_max, err_mean, x_new, a_new, v_new, go, eta, nu

def BP_AMPstep_doubleconv(f,damp, a_old, v_old, go_old, eta_old, nu_old, F, c, wRademacher=False, BPstep=True):
    """
    Performs one step of BP-AMP. Returns the updated variables.
    
    a_old: array M
    v_old: array M
    go_old: array N
    F: array NxM
    c: float
    """ 
    M, N = len(a_old), len(go_old)
    V = np.mean(v_old)
    om = np.dot(F, a_old) - V*go_old
    #print(min(om), max(om), np.isnan(om).sum())
    #if BPstep: 
    nu_nn, norm = f.compute_nu()
    nu = np.array([ (1-damp)*nu_nn[i]/norm[i] + damp*nu_old[i] if norm[i]>0 else nu_old[i]  for i in range(N)])
    #else:
    #nu = nu_old
    #nu= f.compute_nu()
    go, _ = compute_go(nu, om, V, c)
    go = (1-damp)*go +damp*go_old
    #print("go"+f"{go[:10]}")
    eta = damp*eta_old + (1-damp)*compute_eta(om, V, c)
    f.set_delta(eta)
    #print("d"+f"{f.get_delta()[:10]}" + "om"+f"{om[:10]}"+ "V" + f"{V}" + "v_old" + f"{v_old[:10]}")
    #print("nu"+f"{nu[:10]}")
    Av = np.sum(go**2)/M
    Bv = a_old*Av+np.dot(go, F)
    #print(min(go),max(go),np.isnan(go).sum())
    #print("Bv: " + f"{np.dot(go, F)[:10]}" + "Av: " + f"{ (a_old*Av)[:10]}")
    #print(min(Bv),max(Bv),np.isnan(Bv).sum())
    a_new = damp*a_old + (1-damp)*fa(Av, Bv, wRademacher)
    v_new = damp*v_old + (1-damp)*fv(Av, Bv, wRademacher)
    
    if BPstep:
        _, [err_max, err_mean] = f.update(maxit=1000, tol=1e-6, damp=damp) #nu(+1)
    else:
        err_max, err_mean = 1, 1


    chi = np.array([1 if ((nu[i] == 1) or (eta[i]==0)) else 0 if ((eta[i]==1) or (nu[i] == 0)) else nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
    #chi = np.array([nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
    x_new = 2*chi-1
    
    return err_max, err_mean, x_new, a_new, v_new, go, eta, nu

def BP_AMPstep_old(f,damp, a_old, v_old, go_old, eta_old, nu_old, F, c, wRademacher=False):
    """
    Performs one step of BP-AMP. Returns the updated variables.
    
    a_old: array M
    v_old: array M
    go_old: array N
    F: array NxM
    c: float
    """ 
    M, N = len(a_old), len(go_old)
    V = np.mean(v_old)
    om = np.dot(F, a_old) - V*go_old
    #print(min(om), max(om), np.isnan(om).sum())
    nu_nn, norm = f.compute_nu()
    nu = np.array([ (1-damp)*nu_nn[i]/norm[i] + damp*nu_old[i] if norm[i]>0 else nu_old[i]  for i in range(N)])
    #nu= f.compute_nu()
    go, _ = compute_go(nu, om, V, c)
    go = (1-damp)*go +damp*go_old
    #print("go"+f"{go[:10]}")
    eta = damp*eta_old + (1-damp)*compute_eta(om, V, c)
    f.set_delta(eta)
    #print("d"+f"{f.get_delta()[:10]}" + "om"+f"{om[:10]}"+ "V" + f"{V}" + "v_old" + f"{v_old[:10]}")
    #print("nu"+f"{nu[:10]}")
    Av = np.sum(go**2)/M
    Bv = a_old*Av+np.dot(go, F)
    #print(min(go),max(go),np.isnan(go).sum())
    #print("Bv: " + f"{np.dot(go, F)[:10]}" + "Av: " + f"{ (a_old*Av)[:10]}")
    #print(min(Bv),max(Bv),np.isnan(Bv).sum())
    a_new = damp*a_old + (1-damp)*fa(Av, Bv, wRademacher)
    v_new = damp*v_old + (1-damp)*fv(Av, Bv, wRademacher)
    
    err_max, err_mean = f.iterate(damp) #nu(+1)

    chi = np.array([1 if ((nu[i] == 1) or (eta[i]==0)) else 0 if ((eta[i]==1) or (nu[i] == 0)) else nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
    #chi = np.array([nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
    x_new = 2*chi-1
    
    return err_max, err_mean, x_new, a_new, v_new, go, eta, nu

#def BP_AMPstep_switch(f,damp, a_old, v_old, go_old, eta_old, nu_old, F, c, wRademacher=False):
#    """
#    Performs one step of BP-AMP. Returns the updated variables.
#    
#    a_old: array M
#    v_old: array M
#    go_old: array N
#    F: array NxM
#    c: float
#    """ 
#    M, N = len(a_old), len(go_old)
#    V = np.mean(v_old)
#    om = np.dot(F, a_old) - V*go_old
#    eta = damp*eta_old + (1-damp)*compute_eta(om, V, c)
#    f.set_delta(eta)
#    err_max, err_mean = f.iterate(damp) #nu(+1)
#
#    nu= damp*nu_old + (1-damp)*f.compute_nu()
#    go, Znn = compute_go(nu, om, V, c)
#    go = (1-damp)*go + damp*go_old
#
#    Av = np.sum(go**2)/M
#    Bv = a_old*Av+np.dot(go, F)
#    a_new = damp*a_old + (1-damp)*fa(Av, Bv, wRademacher)
#    v_new = damp*v_old + (1-damp)*fv(Av, Bv, wRademacher)
#
#    chi = np.array([1 if ((1-nu[i])*eta[i] < epsilon) else nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
#    x_new = 2*chi-1
#    
#    return err_max, err_mean, x_new, a_new, v_new, go, eta, nu

#def BP_AMPstep(f,damp, a_old, v_old, go_old, F, c, wRademacher=False):
#    """
#    Performs one step of BP-AMP. Returns the updated variables.
#    a_old: array M
#    v_old: array M
#    go_old: array N
#    F: array NxM
#    c: float
#    """
#    M, N = len(a_old), len(go_old)
#    V = np.mean(v_old)
#    om = np.dot(F, a_old) - V*go_old
#    #print(min(om), max(om), np.isnan(om).sum())
#    nu= f.compute_nu()
#    go, Znn = compute_go(nu, om, V, c)
#    #print(go[:10],Znn[:10])
#    eta = compute_eta(om, V, c)
#    f.set_delta(eta)
#    Av = np.sum(go**2)/M
#    Bv = a_old*Av+np.dot(go, F)
#    a_new = fa(Av, Bv, wRademacher)
#    v_new = fv(Av, Bv, wRademacher)
#    err_max, err_mean = f.iterate(damp) #nu(+1)
#    chi = np.array([1 if ((1-nu[i])*eta[i] < epsilon) else nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
#    x_new = 2*chi-1
#    return err_max, err_mean, x_new, a_new, v_new, go
#
#def BP_AMPstep_switch(f,damp, a_old, v_old, go_old, F, c, wRademacher=False):
#    """
#    Performs one step of BP-AMP. Returns the updated variables.
#    a_old: array M
#    v_old: array M
#    go_old: array N
#    F: array NxM
#    c: float
#    """
#    M, N = len(a_old), len(go_old)
#    V = np.mean(v_old)
#    om = np.dot(F, a_old) - V*go_old
#    eta = compute_eta(om, V, c)
#    f.set_delta(eta)
#    err_max, err_mean = f.iterate(damp) #nu(+1)
#    nu= f.compute_nu()
#    go, Znn = compute_go(nu, om, V, c)
#    Av = np.sum(go**2)/M
#    Bv = a_old*Av+np.dot(go, F)
#    a_new = fa(Av, Bv, wRademacher)
#    v_new = fv(Av, Bv, wRademacher)
#    chi = np.array([1 if ((1-nu[i])*eta[i] < epsilon) else nu[i]*(1-eta[i])/(nu[i]*(1-eta[i]) + (1-nu[i])*eta[i]) for i in range(N)])
#    x_new = 2*chi-1
#    return err_max, err_mean, x_new, a_new, v_new, go
