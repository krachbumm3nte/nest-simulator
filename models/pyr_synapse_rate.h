/*
 *  pyr_synapse.h
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

#ifndef PYR_SYNAPSE_RATE_H
#define PYR_SYNAPSE_RATE_H

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"
#include "ring_buffer.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

namespace nest
{

/* BeginUserDocs: synapse, spike-timing-dependent plasticity

Short description
+++++++++++++++++

Synapse type for a plastic synapse after Urbanczik and Senn

Description
+++++++++++

``pyr_synapse`` is a connector to create Urbanczik synapses as defined in
[1]_ that can connect suitable :ref:`multicompartment models
<multicompartment-models>`. In contrast to most STDP models, the synaptic weight
depends on the postsynaptic dendritic potential, in addition to the pre- and
postsynaptic spike timing.

Urbanczik synapses require the archiving of the dendritic membrane potential
which is continuous in time. Therefore they can only be connected to neuron
models that are capable of doing this archiving. So far, the only compatible
model is :doc:`pp_cond_exp_mc_urbanczik <pp_cond_exp_mc_urbanczik>`.

.. warning::

   This synaptic plasticity rule does not take
   :ref:`precise spike timing <sim_precise_spike_times>` into
   account. When calculating the weight update, the precise spike time part
   of the timestamp is ignored.

Parameters
++++++++++

=========   ====   =========================================================
eta         real   Learning rate
tau_Delta   real   Time constant of low pass filtering of the weight change
Wmax        real   Maximum allowed weight
Wmin        real   Minimum allowed weight
=========   ====   =========================================================

All other parameters are stored in the neuron models that are compatible
with the Urbanczik synapse.

Transmits
+++++++++

SpikeEvent

References
++++++++++

.. [1] Urbanczik R. and Senn W (2014). Learning by the dendritic
       prediction of somatic spiking. Neuron, 81:521 - 528.
       https://doi.org/10.1016/j.neuron.2013.11.030

See also
++++++++

stdp_synapse, clopath_synapse, pp_cond_exp_mc_urbanczik

EndUserDocs */

// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template

template < typename targetidentifierT >
class pyr_synapse_rate : public Connection< targetidentifierT >
{

public:
  typedef CommonSynapseProperties CommonPropertiesType;
  typedef Connection< targetidentifierT > ConnectionBase;
  typedef DelayedRateConnectionEvent EventType;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  pyr_synapse_rate()
    : ConnectionBase()
    , weight_( 1.0 )
    , tilde_w( 0 )
    , init_weight_( 0.0 )
    , tau_Delta_( 100.0 )
    , eta_( 0.07 )
    , Wmin_( -1.0 )
    , Wmax_( 1.0 )
    , r_in( 0.0 )
    , u_target( 0.0 )
    , v_dend_target( 0.0 )
  {
  }


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  pyr_synapse_rate( const pyr_synapse_rate& ) = default;
  pyr_synapse_rate& operator=( const pyr_synapse_rate& ) = default;

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay;
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, ConnectorModel& cm );


  double phi( double x );

  // class ConnTestDummyNode : public ConnTestDummyNodeBase
  // {
  // public:
  //   // Ensure proper overriding of overloaded virtual functions.
  //   // Return values from functions are ignored.
  //   using ConnTestDummyNodeBase::handles_test_event;
  //   port
  //   handles_test_event( DelayedRateConnectionEvent&, rport ) override
  //   {
  //     std::cout << "synapse handles" << std::endl;
  //     return 0;
  //   }
  // };

  void
  check_connection( Node& s, Node& t, rport receptor_type, const CommonPropertiesType& )
  {
    EventType ge;
    s.sends_secondary_event( ge );
    ge.set_sender( s );

    Connection< targetidentifierT >::target_.set_rport( t.handles_test_event( ge, receptor_type ) );
    Connection< targetidentifierT >::target_.set_target( &t );
    t.register_stdp_connection( -1 - get_delay(), get_delay() );

  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }


  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param t The thread on which this connection is stored.
   * \param cp Common properties object, containing the stdp parameters.
   */
  void
  send( Event& e, thread t, const CommonSynapseProperties& )
  {
    Node* target = get_target( t );
    nest::rate_neuron_pyr* target_pyr = static_cast< nest::rate_neuron_pyr* >( target );
    
    nest::DelayedRateConnectionEvent& del_event = static_cast< nest::DelayedRateConnectionEvent& >( e );
    int rport = get_rport();
    double delta_tilde_w;
    double dend_error = 0;
    double V_W_star = 0;

    u_target = target->get_V_m( 0 );
    //std::cout << "u_tgt: " << u_target << ", w: " << weight_ << std::endl;
    if ( rport == 1 )
    {
      double const g_L = target->get_g_L( 0 );
      double g_D = target->get_g( 1 );
      double g_A = target->get_g( 2 );
      V_W_star = ( g_D * v_dend_target ) / ( g_L + g_D + g_A );
      dend_error = ( target_pyr->P_.pyr_params.phi( u_target ) - target_pyr->P_.pyr_params.phi( V_W_star ) );
    }
    else if ( rport == 2 )
    {
      dend_error = -v_dend_target;
    }
    else if ( rport == 3 )
    {
      dend_error = ( target_pyr->P_.pyr_params.phi( u_target ) - target_pyr->P_.pyr_params.phi( v_dend_target ) );
    }
    delta_tilde_w = -tilde_w + dend_error * r_in;
    // TODO: generalize 0.1 to delta_t
    tilde_w = tilde_w + 0.1 * ( delta_tilde_w / tau_Delta_ );
    weight_ = weight_ + 0.1 * eta_ * tilde_w;


    if ( weight_ > Wmax_ )
    {
      weight_ = Wmax_;
    }
    else if ( weight_ < Wmin_ )
    {
      weight_ = Wmin_;
    }
    it = del_event.begin();
    //std::cout << r_in << std::endl;
    r_in = del_event.get_coeffvalue( it );
    it--;
    v_dend_target = target->get_V_m( rport );
    //const size_t buffer_size = kernel().connection_manager.get_min_delay();

    //std::vector< double > rate_vec( buffer_size, 0.0 );
    //rate_vec[ 0 ] = r_in;
    //del_event.set_coeffarray( rate_vec );
    del_event.set_receiver( *target );
    del_event.set_delay_steps( get_delay_steps() );
    del_event.set_weight( weight_ );
    del_event.set_rport( rport );
    del_event();
  }

private:
  // data members of each connection
  double weight_;
  double tilde_w;
  double init_weight_;
  double tau_Delta_;
  double eta_;
  double Wmin_;
  double Wmax_;
  double r_in;
  double u_target;
  double v_dend_target;
  std::vector< unsigned int >::iterator it;
};


template < typename targetidentifierT >
void
pyr_synapse_rate< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::tau_Delta, tau_Delta_ );
  def< double >( d, names::eta, eta_ );
  def< double >( d, names::Wmin, Wmin_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< long >( d, names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
pyr_synapse_rate< targetidentifierT >::set_status( const DictionaryDatum& d, ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::tau_Delta, tau_Delta_ );
  updateValue< double >( d, names::eta, eta_ );
  updateValue< double >( d, names::Wmin, Wmin_ );
  updateValue< double >( d, names::Wmax, Wmax_ );

  init_weight_ = weight_;
}

} // of namespace nest

#endif // of #ifndef PYR_SYNAPSE_RATE_H
