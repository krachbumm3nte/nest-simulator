/*
 *  pyr_archiving_node.cpp
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

#ifndef PYR_ARCHIVING_NODE_IMPL_H
#define PYR_ARCHIVING_NODE_IMPL_H

#include "pyr_archiving_node.h"

// Includes from nestkernel:
#include "kernel_manager.h"

// Includes from sli:
#include "dictutils.h"
#include "stdio.h"

namespace nest
{

// member functions for PyrArchivingNode
template < class pyr_parameters >
nest::PyrArchivingNode< pyr_parameters >::PyrArchivingNode()
  : ArchivingNode()
{
}

template < class pyr_parameters >
nest::PyrArchivingNode< pyr_parameters >::PyrArchivingNode( const PyrArchivingNode& n )
  : ArchivingNode( n )
{
}

template < class pyr_parameters >
void
nest::PyrArchivingNode< pyr_parameters >::get_status( DictionaryDatum& d ) const
{
  ArchivingNode::get_status( d );
}

template < class pyr_parameters >
void
nest::PyrArchivingNode< pyr_parameters >::set_status( const DictionaryDatum& d )
{
  ArchivingNode::set_status( d );
}

template < class pyr_parameters >
void
nest::PyrArchivingNode< pyr_parameters >::get_urbanczik_history( double t1,
  double t2,
  std::deque< histentry_extended >::iterator* start,
  std::deque< histentry_extended >::iterator* finish,
  int comp )
{
  *finish = pyr_history_[ comp - 1 ].end();
  if ( pyr_history_[ comp - 1 ].empty() )
  {
    *start = *finish;
    return;
  }
  else
  {
    std::deque< histentry_extended >::iterator runner = pyr_history_[ comp - 1 ].begin();
    // To have a well defined discretization of the integral, we make sure
    // that we exclude the entry at t1 but include the one at t2 by subtracting
    // a small number so that runner->t_ is never equal to t1 or t2.
    while ( ( runner != pyr_history_[ comp - 1 ].end() ) and runner->t_ - 1.0e-6 < t1 )
    {
      ++runner;
    }
    *start = runner;
    while ( ( runner != pyr_history_[ comp - 1 ].end() ) and runner->t_ - 1.0e-6 < t2 )
    {
      ( runner->access_counter_ )++;
      ++runner;
    }
    *finish = runner;
  }
}

template < class pyr_parameters >
void
nest::PyrArchivingNode< pyr_parameters >::write_urbanczik_history( Time const& t_sp,
  double V_W, 
  double V_SOM,
  int comp )
{
  const double t_ms = t_sp.get_ms();

  //const double g_D = pyr_params->g_conn[ pyr_parameters::SOMA ];
  const double g_L = pyr_params->g_L[ pyr_parameters::SOMA ];
  //const double E_L = pyr_params->E_L[ pyr_parameters::SOMA ];
  double V_W_star = 0.0;

  double comp_deviation = 0.0;

  if (comp == 0) {
    std::cout << "pyr history written for somatic compartment";
  } else if (comp == 1) {
    // basal compartment
    const double g_b = pyr_params->g_conn[ pyr_parameters::BASAL ];
    double g_a = pyr_params->g_conn[ pyr_parameters::APICAL_LAT ];

    V_W_star = ( g_b * V_W ) / ( g_L + g_b + g_a);
    // comp_deviation =  (pyr_params->phi( V_SOM ) - pyr_params->phi( V_W_star )) * Time::get_resolution().get_ms();
    comp_deviation =  (pyr_params->phi( V_SOM ) - pyr_params->phi( V_W_star ));
  } else if (comp == 2) {
    // apical compartment for lateral interneuron-pyr connections
    // TODO: is E_L a legitimate placeholder vor v_rest?
    comp_deviation = - V_W;
    // comp_deviation = -V_W * Time::get_resolution().get_ms();
  }
  // } else if (comp == 2) {
    // apical compartment for top-down pyr-pyr connections
    // TODO: top-down synapses require presynpatic factors twice
    // in the synapse, calculation for this compartment should be <this * r_pre - weight * r_pre^2>
    // comp_deviation = (V_SOM - pyr_params->phi( V_W )) * Time::get_resolution().get_ms();


  if ( n_incoming_ )
  {
    // prune all entries from history which are no longer needed
    // except the penultimate one. we might still need it.
    while ( pyr_history_[ comp - 1 ].size() > 1 )
    {
      // This is a disgusting workaround for the issue that archiving_node.access_counter_ is unable to differentiate
      // between compartments, causing the history of multi-compartment models to increase continuously.
      size_t access_counter = 0;
      for (int i = 0; i < 2; i++) {
        access_counter += pyr_history_[i].front().access_counter_;
      }

      if ( access_counter >= n_incoming_ )
      {
        for (int i = 0; i < 2; i++) {
          pyr_history_[ i ].pop_front();
        }
      }
      else
      {
        break;
      }
    }

    // TODO: do we keep the h() term?
    double dPI = comp_deviation; // * pyr_params->h( V_W_star );
    pyr_history_[ comp - 1 ].push_back( histentry_extended( t_ms, dPI, 0 ) );
  }
}

} // of namespace nest

#endif