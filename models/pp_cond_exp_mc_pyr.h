/*
 *  pp_cond_exp_mc_pyr.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PP_COND_EXP_MC_PYR_H
#define PP_COND_EXP_MC_PYR_H

// Generated includes:
#include "config.h"

#ifdef HAVE_GSL

// C++ includes:
#include <vector>

// C includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

// Includes from nestkernel:
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "pyr_archiving_node.h"
#include "pyr_archiving_node_impl.h"
#include "random_generators.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

// Includes from sli:
#include "dictdatum.h"
#include "name.h"

namespace nest
{
/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 */
extern "C" int pp_cond_exp_mc_pyr_dynamics( double, const double*, double*, void* );

/** @BeginDocumentation
Name: pp_cond_exp_mc_pyr_parameters - Helper class for pp_cond_exp_mc_pyr

Description:
``pp_cond_exp_mc_pyr_parameters`` is a helper class for the ``pp_cond_exp_mc_pyr`` neuron model
that contains all parameters of the model that are needed to compute the weight changes of a
connected ``pyr_synapse`` in the base class PyrArchivingNode.

Author: Jonas Stapmanns, David Dahmen, Jan Hahne, Johannes Gille

SeeAlso: pp_cond_exp_mc_pyr
*/
class pp_cond_exp_mc_pyr_parameters
{
  friend class pp_cond_exp_mc_pyr;
  friend class PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >;

private:
  //! Compartments, NCOMP is number
  enum Compartments_
  {
    SOMA = 0,
    BASAL,
    APICAL_LAT,
    // APICAL_TD,
    NCOMP
  };

  double phi_max; //!< Parameter of the rate function
  double gamma;   //!< Parameter of the rate function
  double beta;    //!< Parameter of the rate function
  double theta;   //!< Parameter of the rate function
  double h( double u );

  int curr_target_id; // target neuron for a singular current synapse
  Node* curr_target;
  double lambda_curr;
  bool use_phi;
  bool latent_equilibrium;

public:
  // The pyr parameters need to be public within this class as they are passed to the GSL solver
  double tau_m;
  double phi( double u );

  double g_conn[ NCOMP ]; //!< Conductances connecting compartments in nS
  double g_L[ NCOMP ];    //!< Leak Conductance in nS
  double C_m[ NCOMP ];    //!< Capacity of the membrane in pF
  double E_L[ NCOMP ];    //!< Reversal Potential in mV
};

/* BeginUserDocs: neuron, point process, conductance-based

Short description
+++++++++++++++++

Multi-compartment point process neuron with conductance-based synapses

Description
+++++++++++

pp_cond_exp_mc_pyr is an implementation of a multi-compartment spiking point
process neuron with conductance-based synapses based on the
pp_cond_exp_mc_urbanczik model

The model has three compartments: soma, basal, and apical_lat. A possible
extension to a fourth compartment for top-down inputs (apical_td) is included in
the code, but will not be compiled by default. Each compartment can receive
spike events and current input from a current generator. Additionally, an
external (rheobase) current can be set for each compartment.

Synapses, including those for injection external currents, are addressed through
the receptor types given in the receptor_types entry of the state dictionary.


.. _multicompartment-models:

.. note::

   The neuron model uses standard units of NEST instead of the unitless
   quantities used in [1]_.

.. note::

   All parameters that occur for both compartments are stored as C arrays, with
   index 0 being soma.

Sends
+++++

SpikeEvent

Receives
++++++++

SpikeEvent, CurrentEvent, DataLoggingRequest

References
++++++++++

.. [1] R. Urbanczik, W. Senn (2014). Learning by the Dendritic Prediction of
       Somatic Spiking. Neuron, 81, 521 - 528.

See also
++++++++

urbanczik_synapse

EndUserDocs */

class pp_cond_exp_mc_pyr : public PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >
{

  // Boilerplate function declarations --------------------------------

public:
  pp_cond_exp_mc_pyr();
  pp_cond_exp_mc_pyr( const pp_cond_exp_mc_pyr& );
  ~pp_cond_exp_mc_pyr();

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using Node::handle;
  using Node::handles_test_event;

  port send_test_event( Node&, rport, synindex, bool ) override;

  void handle( SpikeEvent& ) override;
  void handle( CurrentEvent& ) override;
  void handle( DataLoggingRequest& ) override;
  void handle( DelayedRateConnectionEvent& ) override;

  port handles_test_event( DelayedRateConnectionEvent&, rport ) override;
  port handles_test_event( SpikeEvent&, rport ) override;
  port handles_test_event( CurrentEvent&, rport ) override;
  port handles_test_event( DataLoggingRequest&, rport ) override;

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  void init_buffers_() override;
  void pre_run_hook() override;
  void update( Time const&, const long, const long ) override;

  // Enumerations and constants specifying structure and properties ----

  //! Compartments, NCOMP is number
  enum Compartments_
  {
    SOMA = 0,
    BASAL,
    APICAL_LAT,
    // APICAL_TD,
    NCOMP
  };

  /**
   * Minimal spike receptor type.
   * @note Start with 1 so we can forbid port 0 to avoid accidental
   *       creation of connections with no receptor type set.
   */
  static const port MIN_SPIKE_RECEPTOR = 1;

  /**
   * Spike receptors.
   */
  enum SpikeSynapseTypes
  {
    S_SOMA = MIN_SPIKE_RECEPTOR,
    S_BASAL,
    S_APICAL_LAT,
    // S_APICAL_TD,
    SUP_SPIKE_RECEPTOR
  };

  static const size_t NUM_SPIKE_RECEPTORS = SUP_SPIKE_RECEPTOR - MIN_SPIKE_RECEPTOR;

  /**
   * Minimal current receptor type.
   *  @note Start with SUP_SPIKE_RECEPTOR to avoid any overlap and
   *        accidental mix-ups.
   */
  static const port MIN_CURR_RECEPTOR = SUP_SPIKE_RECEPTOR;

  /**
   * Current receptors.
   */
  enum CurrentSynapseTypes
  {
    I_SOMA = MIN_CURR_RECEPTOR,
    I_BASAL,
    I_APICAL_LAT,
    // I_APICAL_TD,
    SUP_CURR_RECEPTOR
  };

  static const size_t NUM_CURR_RECEPTORS = SUP_CURR_RECEPTOR - MIN_CURR_RECEPTOR;

  // Friends --------------------------------------------------------

  friend int pp_cond_exp_mc_pyr_dynamics( double, const double*, double*, void* );

  friend class RecordablesMap< pp_cond_exp_mc_pyr >;
  friend class UniversalDataLogger< pp_cond_exp_mc_pyr >;


  // Parameters ------------------------------------------------------

  /**
   * Independent parameters of the model.
   * These parameters must be passed to the iteration function that
   * is passed to the GSL ODE solvers. Since the iteration function
   * is a C++ function with C linkage, the parameters can be stored
   * in a C++ struct with member functions, as long as we just pass
   * it by void* from C++ to C++ function. The struct must be public,
   * though, since the iteration function is a function with C-linkage,
   * whence it cannot be a member function of pp_cond_exp_mc_pyr.
   * @note One could achieve proper encapsulation by an extra level
   *       of indirection: Define the iteration function as a member
   *       function, plus an additional wrapper function with C linkage.
   *       Then pass a struct containing a pointer to the node and a
   *       pointer-to-member-function to the iteration function as void*
   *       to the wrapper function. The wrapper function can then invoke
   *       the iteration function on the node (Stroustrup, p 418). But
   *       this appears to involved, and the extra indirections cost.
   */
  struct Parameters_
  {
    double t_ref;        //!< Refractory period in ms
    double I_e[ NCOMP ]; //!< Constant Current in pA

    pp_cond_exp_mc_pyr_parameters pyr_params;

    /** Dead time in ms. */
    double dead_time_;

    Parameters_();                                //!< Sets default parameter values
    Parameters_( const Parameters_& );            //!< needed to copy C-arrays
    Parameters_& operator=( const Parameters_& ); //!< needed to copy C-arrays

    void get( DictionaryDatum& ) const;           //!< Store current values in dictionary
    void set( const DictionaryDatum& );           //!< Set values from dictionary
  };


  // State variables  ------------------------------------------------------

  /**
   * State variables of the model.
   * @note Copy constructor required because of C-style array.
   */
public:
  struct State_
  {

    /**
     * Elements of state vector.
     * For the multicompartmental case here, these are offset values.
     * The state variables are stored in contiguous blocks for each
     * compartment, beginning with the soma.
     */
    enum StateVecElems_
    {
      V_M = 0,
      V_forw,
      STATE_VEC_COMPS
    };

    //! total size of state vector
    static const size_t STATE_VEC_SIZE = STATE_VEC_COMPS * NCOMP;

    //! neuron state, must be C-array for GSL solver
    double y_[ STATE_VEC_SIZE ];
    int r_;                       //!< number of refractory steps remaining

    State_( const Parameters_& ); //!< Default initialization
    State_( const State_& );

    State_& operator=( const State_& );

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum&, const Parameters_& );

    /**
     * Compute linear index into state array from compartment and element.
     * @param comp compartment index
     * @param elem elemet index
     * @note compartment argument is not of type Compartments_, since looping
     *       over enumerations does not work.
     */
    static size_t
    idx( size_t comp, StateVecElems_ elem )
    {
      assert( comp * STATE_VEC_COMPS + elem < STATE_VEC_SIZE );
      return comp * STATE_VEC_COMPS + elem;
    }
  };

  double
  get_V_m( int comp )
  {
    return S_.y_[ S_.idx( comp, State_::V_M ) ];
  }

private:
  // Internal buffers --------------------------------------------------------

  /**
   * Buffers of the model.
   */
  struct Buffers_
  {
    Buffers_( pp_cond_exp_mc_pyr& ); //!< Sets buffer pointers to 0
    //! Sets buffer pointers to 0
    Buffers_( const Buffers_&, pp_cond_exp_mc_pyr& );

    //! Logger for all analog data
    UniversalDataLogger< pp_cond_exp_mc_pyr > logger_;

    /** buffers and sums up incoming spikes/currents
     *  @note Using STL vectors here to ensure initialization.
     */
    std::vector< RingBuffer > spikes_;
    std::vector< RingBuffer > currents_;

    /** GSL ODE stuff */
    gsl_odeiv_step* s_;    //!< stepping function
    gsl_odeiv_control* c_; //!< adaptive stepsize control function
    gsl_odeiv_evolve* e_;  //!< evolution function
    gsl_odeiv_system sys_; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double step_;            //!< step size in ms
    double IntegrationStep_; //!< current integration time step, updated by GSL

    /**
     * Input currents injected by CurrentEvent.
     * This variable is used to transport the current applied into the
     * _dynamics function computing the derivative of the state vector.
     * It must be a part of Buffers_, since it is initialized once before
     * the first simulation, but not modified before later Simulate calls.
     */
    double I_stim_[ NCOMP ]; //!< External Stimulus in pA
  };

  // Internal variables ---------------------------------------------

  /**
   * Internal variables of the model.
   */
  struct Variables_
  {
    int RefractoryCounts_;

    double h_;                          //!< simulation time step in ms
    RngPtr rng_;                        //!< random number generator of my own thread
    poisson_distribution poisson_dist_; //!< poisson distribution
  };

  // Access functions for UniversalDataLogger -------------------------------

  /**
   * Read out state vector elements, used by UniversalDataLogger
   * First template argument is component "name", second compartment "name".
   */
  template < State_::StateVecElems_ elem, Compartments_ comp >
  double
  get_y_elem_() const
  {
    return S_.y_[ S_.idx( comp, elem ) ];
  }

  //! Read out number of refractory steps, used by UniversalDataLogger
  double
  get_r_() const
  {
    return Time::get_resolution().get_ms() * S_.r_;
  }


  // Data members ----------------------------------------------------
public:
  Parameters_ P_;
  State_ S_;
  Variables_ V_;
  Buffers_ B_;

  //! Table of compartment names
  static std::vector< Name > comp_names_;

  //! Dictionary of receptor types, leads to seg fault on exit, see #328
  // static DictionaryDatum receptor_dict_;

  //! Mapping of recordables names to access functions
  static RecordablesMap< pp_cond_exp_mc_pyr > recordablesMap_;
};


// Inline functions of pp_cond_exp_mc_pyr_parameters
const double phi_thresh = 15;

inline double
pp_cond_exp_mc_pyr_parameters::phi( double u )
{
  if ( use_phi )
  {
    if ( u < -phi_thresh )
    {
      u = 0;
    }
    else if ( u > phi_thresh )
    {
      return u * gamma;
    }
    return gamma * log( 1 + exp( beta * ( u - theta ) ) );
  }
  else
  {
    return u * gamma;
  }
}

// Inline functions of pp_cond_exp_mc_pyr
inline port
pp_cond_exp_mc_pyr::send_test_event( Node& target, rport receptor_type, synindex, bool )
{
  SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline port
pp_cond_exp_mc_pyr::handles_test_event( SpikeEvent&, rport receptor_type )
{
  if ( receptor_type < MIN_SPIKE_RECEPTOR or receptor_type >= SUP_SPIKE_RECEPTOR )
  {
    if ( receptor_type < 0 or receptor_type >= SUP_CURR_RECEPTOR )
    {
      throw UnknownReceptorType( receptor_type, get_name() );
    }
    else
    {
      throw IncompatibleReceptorType( receptor_type, get_name(), "SpikeEvent" );
    }
  }
  return receptor_type - MIN_SPIKE_RECEPTOR;
}

inline port
pp_cond_exp_mc_pyr::handles_test_event( DelayedRateConnectionEvent&, rport receptor_type )
{
  if ( receptor_type < MIN_SPIKE_RECEPTOR || receptor_type >= SUP_SPIKE_RECEPTOR )
  {
    if ( receptor_type < 0 || receptor_type >= SUP_CURR_RECEPTOR )
    {
      throw UnknownReceptorType( receptor_type, get_name() );
    }
    else
    {
      throw IncompatibleReceptorType( receptor_type, get_name(), "DelayedRateConnectionEvent" );
    }
  }
  return receptor_type - MIN_SPIKE_RECEPTOR;
}

inline port
pp_cond_exp_mc_pyr::handles_test_event( CurrentEvent&, rport receptor_type )
{
  if ( receptor_type < MIN_CURR_RECEPTOR or receptor_type >= SUP_CURR_RECEPTOR )
  {
    if ( receptor_type >= 0 and receptor_type < MIN_CURR_RECEPTOR )
    {
      throw IncompatibleReceptorType( receptor_type, get_name(), "CurrentEvent" );
    }
    else
    {
      throw UnknownReceptorType( receptor_type, get_name() );
    }
  }
  return receptor_type - MIN_CURR_RECEPTOR;
}

inline port
pp_cond_exp_mc_pyr::handles_test_event( DataLoggingRequest& dlr, rport receptor_type )
{
  if ( receptor_type != 0 )
  {
    if ( receptor_type < 0 or receptor_type >= SUP_CURR_RECEPTOR )
    {
      throw UnknownReceptorType( receptor_type, get_name() );
    }
    else
    {
      throw IncompatibleReceptorType( receptor_type, get_name(), "DataLoggingRequest" );
    }
  }
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
pp_cond_exp_mc_pyr::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >::get_status( d );

  ( *d )[ names::recordables ] = recordablesMap_.get_list();

  /**
   * @todo dictionary construction should be done only once for
   * static member in default c'tor, but this leads to
   * a seg fault on exit, see #328
   */
  DictionaryDatum receptor_dict_ = new Dictionary();
  ( *receptor_dict_ )[ names::soma ] = S_SOMA;
  ( *receptor_dict_ )[ names::soma_curr ] = I_SOMA;

  ( *receptor_dict_ )[ names::basal ] = S_BASAL;
  ( *receptor_dict_ )[ names::basal_curr ] = I_BASAL;

  ( *receptor_dict_ )[ names::apical_lat ] = S_APICAL_LAT;
  ( *receptor_dict_ )[ names::apical_lat_curr ] = I_APICAL_LAT;

  // ( *receptor_dict_ )[ names::apical_td ] = S_APICAL_TD;
  // ( *receptor_dict_ )[ names::apical_td_curr ] = I_APICAL_TD;

  ( *d )[ names::receptor_types ] = receptor_dict_;
}

inline void
pp_cond_exp_mc_pyr::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >::set_status( d );
  PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >::clear_history();
  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace


#endif // HAVE_GSL
#endif // PP_COND_EXP_MC_PYR_H
