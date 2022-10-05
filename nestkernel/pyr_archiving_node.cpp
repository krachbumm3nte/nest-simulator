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

#include "pyr_archiving_node.h"

// Includes from nestkernel:
#include "kernel_manager.h"

// Includes from sli:
#include "dictutils.h"
#include "stdio.h"
#include<iostream>
#include<fstream>

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
  // remove the oldest entries from pyr_history_ to prevent runaway computing time.
  //if (pyr_history_[ comp - 1].size() > 7000) {
    //std::cout << "purging compartment " << comp-1 << ", conns: " << n_incoming_ << "\n";
    //pyr_history_[ comp - 1].erase(pyr_history_[ comp - 1 ].begin(), pyr_history_[ comp - 1 ].begin() + 500);
  //}


  //*finish = pyr_history_[ comp - 1 ].end();
  
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
    int counter = 0;
    while ( ( runner != pyr_history_[ comp - 1 ].end() ) && ( runner->t_ - 1.0e-6 < t1 ) )
    {
      ++counter;
      //std::cout << runner->access_counter_ << "\n";
      ++runner;
    }
    //std::cout << counter << std::endl;
    
    *start = runner;

    while ( ( runner != pyr_history_[ comp - 1 ].end() ) && ( runner->t_ - 1.0e-6 < t2 ) )
    {
      runner->access_counter_++;
      //std::cout << runner->access_counter_ << "\n";

      ++runner;
    }
    *finish = runner;
  }
}

template < class pyr_parameters >
void
nest::PyrArchivingNode< pyr_parameters >::write_urbanczik_history( Time const& t_sp,
  double V_W,
  int n_spikes,
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
    if (g_a == 0) {
      g_a = 1; // avoid zero division for interneurons where the apical compartment is silenced.
    }
    V_W_star = ( g_b * V_W ) / ( g_L * g_b * g_a);
    comp_deviation = n_spikes - pyr_params->phi( V_W_star ) * Time::get_resolution().get_ms();
  } else if (comp == 2) {
    // apical compartment for top-down pyr-pyr connections
    // TODO: top-down synapses require presynpatic factors twice
    // in the synapse, calculation for this compartment should be <this * r_pre - weight * r_pre^2>
    comp_deviation = n_spikes - pyr_params->phi( V_W ) * Time::get_resolution().get_ms();
  } else if (comp == 3) {
    // apical compartment for lateral interneuron-pyr connections
    // TODO: is E_L a legitimate placeholder vor v_rest?
    comp_deviation = pyr_params->E_L[0] - V_W;
  }

  if ( n_incoming_ )
  {
    // prune all entries from history which are no longer needed
    // except the penultimate one. we might still need it.
    int s = pyr_history_[ comp - 1 ].size();
    while ( pyr_history_[ comp - 1 ].size() > 1 )
    {
      if ( pyr_history_[ comp - 1 ].front().access_counter_ >= n_incoming_ )
      {
        pyr_history_[ comp - 1 ].pop_front();
      }
      else
      {
        break;
      }
    }

    s -= pyr_history_[ comp - 1 ].size();
    //if (s > 0) {
      //std::cout << "pruned compartment " << comp - 1 << " by " << s << "\n"; 
    //}


    // TODO: do we keep the h() term?
    double dPI = comp_deviation; // * pyr_params->h( V_W_star );
    pyr_history_[ comp - 1 ].push_back( histentry_extended( t_ms, dPI, 0 ) );
  }
}

} // of namespace nest
