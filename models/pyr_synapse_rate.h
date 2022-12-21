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

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  pyr_synapse_rate();


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

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp common properties of all synapses (empty).
   */
  void send( Event& e, thread t, const CommonSynapseProperties& cp );


  double phi( double x );

  class ConnTestDummyNode : public ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using ConnTestDummyNodeBase::handles_test_event;
    port
    handles_test_event( SpikeEvent&, rport ) override
    {
      return invalid_port;
    }
  };

  void
  check_connection( Node& s, Node& t, rport receptor_type, const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;

    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );

    t.register_stdp_connection( -1 - get_delay(), get_delay() );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
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
  size_t counter;
};

template < typename targetidentifierT >
inline double
pyr_synapse_rate< targetidentifierT >::phi( double x )
{
  return 1 / ( 1 + exp( -x ) );
  // return log( 1 + exp( x ) );
}


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
pyr_synapse_rate< targetidentifierT >::send( Event& e, thread t, const CommonSynapseProperties& )
{

  Node* target = get_target( t );
  nest::pp_cond_exp_mc_pyr* target_pyr = static_cast<nest::pp_cond_exp_mc_pyr*>(target);

  Node* sender = kernel().node_manager.get_node_or_proxy( e.retrieve_sender_node_id_from_source_table() );
  nest::pp_cond_exp_mc_pyr* sender_pyr = static_cast<nest::pp_cond_exp_mc_pyr*>(sender);
  double U_sender = sender_pyr->get_V_m( 0 );
  int rport = get_rport();
  double V_dend = target->get_V_m( rport );
  double delta_tilde_w;
  double rate_sender = sender_pyr->P_.pyr_params.phi( U_sender );
  double dend_error = 0;
  double V_W_star = 0;

  if ( rport == 1 )
  {
    double U_target = target->get_V_m( 0 );
    double const g_L = target->get_g_L( 0 );
    double g_D = target->get_g( 1 );
    double g_A = target->get_g( 2 );
    V_W_star =  ( g_D * V_dend ) / ( g_L + g_D + g_A );
    dend_error = ( target_pyr->P_.pyr_params.phi( U_target ) - target_pyr->P_.pyr_params.phi( V_W_star ));
    // if ( sender->get_node_id() == 3 and target->get_node_id() == 5 )
    // {
    //   std::cout << "vars: " << U_target << "," << V_dend << ", " << U_sender << std::endl;
    // }
  }
  else if ( rport == 2)
  {
    dend_error = -V_dend;
  }
  else if (rport == 3) 
  {
    //TODO: this is unverified as of yet, but would enable learning of feedback pyr-pyr weights
    double U_target = target->get_V_m( 0 );
    double V_W_star = weight_ * target_pyr->P_.pyr_params.phi(U_sender);
    dend_error = (target_pyr->P_.pyr_params.phi( U_target ) - target_pyr->P_.pyr_params.phi( V_W_star ));
    rport -= 1; // send all top-down signals to the apical compartment by changing rport
  }
  delta_tilde_w = -tilde_w + dend_error * rate_sender;
  // std::cout << "a: " << rport << ", " << tilde_w << ", " << V_dend << ", " << delta_tilde_w << std::endl;
  //  TODO: generalize delta_t
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


  // std::cout << "syn: " << rate_sender << ", " << weight_ << ", " << tilde_w << ", " << delta_tilde_w << ", " <<
  // dend_error << ", " << V_W_star << std::endl;
  counter = 0;
  //if ( sender->get_node_id() == 3 and target->get_node_id() == 5 )
  //{
  //  std::cout << "syn: " << weight_ << ", " << tilde_w << ", " << delta_tilde_w << std::endl;
  //}
  e.set_receiver( *target );
  e.set_weight( weight_ );
  e.set_rport( rport );
  e();

}


template < typename targetidentifierT >
pyr_synapse_rate< targetidentifierT >::pyr_synapse_rate()
  : ConnectionBase()
  , weight_( 1.0 )
  , tilde_w( 0 )
  , init_weight_( 0.0 )
  , tau_Delta_( 100.0 )
  , eta_( 0.07 )
  , Wmin_( -1.0 )
  , Wmax_( 1.0 )
  , counter( 0 )
{
}

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

#endif // of #ifndef URBANCZIK_SYNAPSE_H
