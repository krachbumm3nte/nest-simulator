/*
 *  pyr_archiving_node.h
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

#ifndef PYR_ARCHIVING_NODE_H
#define PYR_ARCHIVING_NODE_H

// C++ includes:
#include <deque>

// Includes from nestkernel:
#include "archiving_node.h"
#include "histentry.h"
#include "nest_time.h"
#include "nest_types.h"
#include "synaptic_element.h"

// Includes from sli:
#include "dictdatum.h"

namespace nest
{

/**
 * \class PyrArchivingNode
 * a archiving node which additionally archives parameters
 * and buffers needed for the Urbanczik-Senn plasticity rule
 */
template < class pyr_parameters >
class PyrArchivingNode : public ArchivingNode
{

public:
  /**
   * \fn PyrArchivingNode()
   * Constructor.
   */
  PyrArchivingNode();

  /**
   * \fn PyrArchivingNode()
   * Copy Constructor.
   */
  PyrArchivingNode( const PyrArchivingNode& );

  bool
  supports_urbanczik_archiving() const
  {
    return true;
  }

  /**
   * \fn void get_urbanczik_history( double t1, double t2,
   * std::deque<Archiver::histentry>::iterator* start,
   * std::deque<Archiver::histentry>::iterator* finish, int comp )
   * Sets pointer start (finish) to the first (last) entry in pyr_history_[comp]
   * whose time argument is between t1 and t2
   */
  void get_urbanczik_history( double t1,
    double t2,
    std::deque< histentry_extended >::iterator* start,
    std::deque< histentry_extended >::iterator* finish,
    int comp );

  /**
   * \fn double get_C_m( int comp )
   * Returns membrane capacitance
   */
  double get_C_m( int comp );

  /**
   * \fn double get_g_L( int comp )
   * Returns leak conductance g_L
   */
  double get_g_L( int comp );

  /**
   * \fn double get_g( int comp )
   * Returns conductance g connecting compartments
   */
  double get_g( int comp );

  /**
   * \fn double get_tau_L( int comp )
   * Returns time constant tau_L
   */
  double get_tau_L( int comp );

  /**
   * \fn double get_tau_s()
   * Returns synaptic time constant tau_syn
   */
  double get_tau_s( int comp );

protected:
  /**
   * \fn void write_urbanczik_history( Time const& t_sp, double V_W, int n_spikes, int comp ))
   * Writes the history for compartment comp into the buffers.
   */
  void write_urbanczik_history( Time const& t_sp, double V_W, int n_spikes, int comp );

  pyr_parameters* pyr_params;

  void get_status( DictionaryDatum& d ) const;
  void set_status( const DictionaryDatum& d );

private:
  std::deque< histentry_extended > pyr_history_[ pyr_parameters::NCOMP - 1 ];
};

template < class pyr_parameters >
inline double
PyrArchivingNode< pyr_parameters >::get_C_m( int comp )
{
  return pyr_params->C_m;
}

template < class pyr_parameters >
inline double
PyrArchivingNode< pyr_parameters >::get_g_L( int comp )
{
  return pyr_params->g_L[ comp ];
}

template < class pyr_parameters >
inline double
PyrArchivingNode< pyr_parameters >::get_g( int comp )
{
  return pyr_params->g_conn[ comp ];
}

template < class pyr_parameters >
inline double
PyrArchivingNode< pyr_parameters >::get_tau_L( int comp )
{
  //TODO: what to do with this
  return 1 / pyr_params->g_L[ comp ];
}

template < class pyr_parameters >
inline double
PyrArchivingNode< pyr_parameters >::get_tau_s( int comp )
{
  return pyr_params->tau_syn;
}


} // of namespace
#endif
